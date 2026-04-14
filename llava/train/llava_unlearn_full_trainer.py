import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Sampler, DataLoader, Dataset, RandomSampler, SequentialSampler, Subset
from peft import get_peft_model, PeftConfig, PeftModelForCausalLM, PeftModel
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import huggingface_hub.utils as hf_hub_utils
import itertools
import re
import copy
import shutil
import deepspeed
import datasets
import json
from accelerate import Accelerator
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from llava.model.builder import load_pretrained_model
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.utils import (
    is_datasets_available,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
    
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
        
                
def get_module_by_path(model, path):
    """
    Retrieve a submodule from the model using a path string.
    For example, path='layers[7].self_attn.q_proj.lora_A'
    """
    current_module = model
    for part in path.split('.'):
        if '[' in part and ']' in part:
            module_name, index = part.split('[')
            index = int(index.rstrip(']'))
            current_module = getattr(current_module, module_name)[index]
        else:
            current_module = getattr(current_module, part)
    return current_module

def get_base_model(model):
    while hasattr(model, 'module'):
        model = model.module
    return model
                
class LLaVAUnlearnTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        forget_dataset: Optional[Dataset] = None,  # new parameter for forget dataset
        frozen_model: Union[PreTrainedModel, nn.Module] = None, # new parameter for frozen model
    ):
        """
        Initialize the LLaVAUnlearnTrainer.

        Args:
            retain_dataset (Optional[Dataset]): The dataset to be used for retention learning.
            Other arguments are the same as those in the parent Trainer class.
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        try:
            base_model = self.model.get_model()
        except AttributeError:
            base_model = self.model
        
        # create the frozen_model
        self.frozen_model = frozen_model
        self.frozen_model.to(self.model.device)
        self.frozen_model = self.frozen_model.to(dtype=self.model.dtype)
        
        try:
            frozen_base_model = self.frozen_model.get_model()
        except AttributeError:
            frozen_base_model = self.frozen_model
        
        if self.args.unlearn_type == "rmu":
            # Check if base_model contains 'layers' and get the number of layers
            if not hasattr(base_model, 'layers'):
                raise AttributeError("The base model is missing the 'layers' attribute.")        
            total_layers = len(base_model.layers)
            # print(f"model has {total_layers} layers")
            if not (0 <= self.args.rmu_layer_id < total_layers):
                raise ValueError(f"layer_id {self.args.rmu_layer_id} is out of range. The model has {total_layers} layers.")

            # Initialize LoRA module lists
            self.frozen_lora_modules = []
            self.updated_lora_modules = [] 
            # Extract updated_lora_modules
            self.updated_lora_modules = eval('base_model.layers[{layer_id}]'.format(layer_id=self.args.rmu_layer_id))
            self.frozen_lora_modules = eval('frozen_base_model.layers[{layer_id}]'.format(layer_id=self.args.rmu_layer_id))

            # updated parameters: mm_projector, and params_id in layer_ids
            for param in self.model.parameters():
                param.requires_grad = False   
                        
            # updated parameters: mm_projector
            projection = base_model.mm_projector
            for param in projection.parameters():
                param.requires_grad = True  
            
            # updated parameters: params_id in layer_ids            
            for layer_id in self.args.rmu_layer_ids:
                for index, param in enumerate(base_model.layers[layer_id].parameters()):
                    if index in self.args.rmu_param_ids:
                        param.requires_grad = True
            
            for name, param in enumerate(self.model.named_parameters()):
                if param[1].requires_grad:
                    print(param[0])    
            # import ipdb; ipdb.set_trace()
            
        # Define the parameters to be updated, including only the parameters of LoRA modules (i.e., those with requires_grad=True)
        # self.params_to_update = []
        # for param in base_model.layers[7].named_parameters():
        #     print(param[0], param[1].size(), param[1].requires_grad)  
        # import ipdb; ipdb.set_trace()         
        # for name, param in enumerate(base_model.named_parameters()):
        #     if param[1].requires_grad:
        #         print(param[0])
        #         print(f"param.requires_grad={param[1].requires_grad}, shape={param[1].size()}")
        #         # self.params_to_update.append(param)
        # import ipdb; ipdb.set_trace()

        # forget dataset parameters
        self.forget_dataset = forget_dataset
        
        # process forget_dataset & retain_dataset
        if self.forget_dataset:
            logger.info("initialize RMU related parameters")
    
            # generate dataloaders for forget and retain datasets
            self.forget_dataloaders = [self.get_dataloader(self.forget_dataset)]
            # generate iterator for forget and retain dataloaders
            self.forget_iterators = [iter(dl) for dl in self.forget_dataloaders]
            
            if self.args.unlearn_type == "rmu":            
                # generate control_vec for forget_dataset
                # self.control_vectors_list = []
                random_vector = torch.rand(1,1, base_model.config.hidden_size, dtype=self.model.dtype, device=self.model.device)
                control_vec = random_vector / torch.norm(random_vector) * args.rmu_steering_coeff_list[0]
                # self.control_vectors_list.append(control_vec)
                self.control_vector = control_vec
                
                if self.args.rmu_retain_alpha != 0:
                    self.retain_dataset = train_dataset
                    self.retain_dataloaders = [self.get_dataloader(self.retain_dataset)]
                    self.retain_iterators = [iter(dl) for dl in self.retain_dataloaders]
                
            if self.args.unlearn_type == "npo" and self.args.npo_retain_alpha != 0:
                self.retain_dataset = train_dataset
                self.retain_dataloaders = [self.get_dataloader(self.retain_dataset)]
                self.retain_iterators = [iter(dl) for dl in self.retain_dataloaders]

        else:
            self.forget_dataloaders = None
            self.retain_dataloaders = None
            self.forget_iterators = None
            self.retain_iterators = None
            self.control_vectors = None
            
            logger.info("Skil RMU related parameters initialization.")
        
        # create accelerator object
        self.frozen_accelerator = Accelerator(
            dispatch_batches=self.args.dispatch_batches,
            split_batches=self.args.split_batches,
            deepspeed_plugin=self.args.deepspeed_plugin,
        )
        self.frozen_acc_model = self.frozen_accelerator.prepare(self.frozen_model)
            
        # initilizing loss logging
        self.loss_dir = self.args.loss_dir if hasattr(self.args, 'loss_dir') else None
        if self.loss_dir:
            os.makedirs(self.loss_dir, exist_ok=True)
            self.loss_file_path = os.path.join(self.loss_dir, 'loss.json')
            # create or reset loss.json file
            with open(self.loss_file_path, 'w') as f:
                pass  
            logger.info(f"Initialized loss logging at {self.loss_file_path}")
        else:
            logger.warning("No 'loss_dir' specified in TrainingArguments. Loss will not be logged.")
                         
    def _get_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None
        
        if self.args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
            
        # Build the sampler.
        elif self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(train_dataset)

    def get_dataloader(self, dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        If forget_dataloader exists, perform RMU training step; else, use the original training_step.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # if self.retain_dataset is not None and self.frozen_model:
        if self.frozen_model:
            
            model.train()
            # self.frozen_model.eval()
            
            if self.args.unlearn_type == "rmu":
                # initialize loss variables
                total_loss = torch.tensor(0.0, device=model.device)
            
                if self.args.rmu_retain_alpha != 0:
                    frozen_model = self.frozen_acc_model
                
                try:
                    # obtain the froget data, forget_dataloader is defaultly a list with only one element
                    forget_batch = next(self.forget_iterators[0])
                    # process forget_batch
                    forget_inputs = self._prepare_inputs(forget_batch)
                    forget_inputs = {k: v.to(model.device) for k, v in forget_inputs.items()}
                    # calculate unlearn loss
                    unlearn_loss = self.compute_unlearn_loss(model, forget_inputs)          
                    
                    if self.args.rmu_retain_alpha != 0:
                        # obtain the retain data, retain_dataloader is defaultly a list with only one element
                        retain_batch = next(self.retain_iterators[0])
                        # process retain_batch
                        retain_inputs = self._prepare_inputs(retain_batch)
                        retain_inputs = {k: v.to(model.device) for k, v in retain_inputs.items()}   
                        
                        # Compute the retain loss to ensure the model's activations on the retained data remain consistent with the frozen_model
                        # self.test_model(model, frozen_model, retain_inputs)
                        retain_loss = self.compute_retain_loss(model, frozen_model, retain_inputs)
                    
                    else:
                        retain_loss = torch.tensor(0.0, device=model.device)
                    
                    # calculate total loss
                    total_loss += unlearn_loss + retain_loss
                        
                except StopIteration:
                    # if forget_dataloader has finished, resetting iterators.
                    logger.info("forget_dataloader or retain_dataloader exhausted. Resetting iterators.")
                    self.forget_iterators = [iter(dl) for dl in self.forget_dataloaders]
                    # if self.args.rmu_retain_alpha != 0:
                    #     self.retain_iterators = [iter(dl) for dl in self.retain_dataloaders]
                        
                    try:
                        forget_batch = next(self.forget_iterators[0])
                        forget_inputs = self._prepare_inputs(forget_batch)
                        forget_inputs = {k: v.to(model.device) for k, v in forget_inputs.items()}

                        unlearn_loss = self.compute_unlearn_loss(model, forget_inputs)
                        
                        if self.args.rmu_retain_alpha != 0:
                            retain_batch = next(self.retain_iterators[0])
                            retain_inputs = self._prepare_inputs(retain_batch)
                            retain_inputs = {k: v.to(model.device) for k, v in retain_inputs.items()}   
                            
                            retain_loss = self.compute_retain_loss(model, frozen_model, retain_inputs)
                        
                        else:
                            retain_loss = torch.tensor(0.0, device=model.device)
                        
                        total_loss += unlearn_loss + retain_loss

                    except StopIteration:
                        # If there is still no data after reset, skip backpropagation
                        logger.warning("After resetting, forget_dataloader or retain_dataloader is still exhausted. Skipping loss computation.")
                        return torch.tensor(0.0, device=model.device)

                # calculate llava default loss
                inputs = self._prepare_inputs(inputs)     
                llava_loss = self.compute_loss(model, inputs)
                # combine the unlearn_loss, retain_loss, llava_loss  
                loss = total_loss + self.args.rmu_llava_loss_weight * llava_loss
                
                # record loss information to loss.json
                if self.loss_dir:
                    loss_entry = {
                        "step": self.state.global_step,
                        "loss": loss.item(),
                        "llava_loss": llava_loss.item(),
                        "unlearn_loss": unlearn_loss.item(),
                        "retain_loss": retain_loss.item(),
                        "learning_rate": self._get_learning_rate(),
                        "epoch": self.state.epoch
                    }
                    try:
                        with open(self.loss_file_path, 'a') as f:
                            f.write(json.dumps(loss_entry) + '\n\n')  
                    except Exception as e:
                        logger.error(f"Failed to write loss entry to {self.loss_file_path}: {e}")
                        
                if self.args.verbose:
                    print(f"Step {self.state.global_step}: Total Loss={loss}, LLaVA Loss={llava_loss}, Retain Loss={retain_loss}, Unlearn Loss={unlearn_loss}")
                    
            elif self.args.unlearn_type == "npo":
                npo_loss = torch.tensor(0.0, device=model.device)
                
                frozen_model = self.frozen_acc_model
                    
                try:
                    # obtain the froget data, forget_dataloader is defaultly a list with only one element
                    forget_batch = next(self.forget_iterators[0])
                    # process forget_batch
                    forget_inputs = self._prepare_inputs(forget_batch)
                    forget_inputs = {k: v.to(model.device) for k, v in forget_inputs.items()}
                    
                    prepared_forget_inputs = self._prepare_inputs(forget_inputs)
                    
                    forget_loss_current = self.compute_loss(model, prepared_forget_inputs)
                    
                    with torch.no_grad():
                        forget_loss_oracle = self.compute_loss(frozen_model, prepared_forget_inputs)
                    
                    neg_log_ratios = forget_loss_current - forget_loss_oracle
                    
                    forget_npo_loss = self.args.npo_forget_alpha * (-F.logsigmoid(self.args.npo_beta * neg_log_ratios).mean() * 2 / self.args.npo_beta)
                    retain_npo_loss = torch.tensor(0.0, device=model.device)
                    
                    if self.args.npo_retain_alpha != 0:
                        retain_batch = next(self.retain_iterators[0])
                        retain_inputs = self._prepare_inputs(retain_batch)
                        retain_inputs = {k: v.to(model.device) for k, v in retain_inputs.items()}   
                        
                        prepared_retain_inputs = self._prepare_inputs(retain_inputs)
                        
                        retain_loss_current = self.compute_loss(model, prepared_retain_inputs)
                        
                        with torch.no_grad():
                            retain_loss_oracle = self.compute_loss(frozen_model, prepared_retain_inputs)
                        
                        retain_neg_log_ratios = retain_loss_current - retain_loss_oracle
                        
                        retain_npo_loss = self.args.npo_retain_alpha * (F.logsigmoid(self.args.npo_beta * retain_neg_log_ratios).mean() * 2 / self.args.npo_beta)
                    
                except StopIteration:
                    # If there is still no data after reset, skip backpropagation
                    logger.info("forget_dataloader or retain_dataloader exhausted. Resetting iterators.")
                    self.forget_iterators = [iter(dl) for dl in self.forget_dataloaders]
                    
                    if self.args.npo_retain_alpha != 0:
                        self.retain_iterators = [iter(dl) for dl in self.retain_dataloaders]
                        
                    try:
                        # obtain the froget data, forget_dataloader is defaultly a list with only one element
                        forget_batch = next(self.forget_iterators[0])
                        # process forget_batch
                        forget_inputs = self._prepare_inputs(forget_batch)
                        forget_inputs = {k: v.to(model.device) for k, v in forget_inputs.items()}
                        
                        prepared_forget_inputs = self._prepare_inputs(forget_inputs)
                        
                        forget_loss_current = self.compute_loss(model, prepared_forget_inputs)
                        
                        with torch.no_grad():
                            forget_loss_oracle = self.compute_loss(frozen_model, prepared_forget_inputs)
                        
                        neg_log_ratios = forget_loss_current - forget_loss_oracle
                        
                        forget_npo_loss = self.args.npo_forget_alpha * (-F.logsigmoid(self.args.npo_beta * neg_log_ratios).mean() * 2 / self.args.npo_beta)
                        retain_npo_loss = torch.tensor(0.0, device=model.device)
                        
                        if self.args.npo_retain_alpha != 0:
                            retain_batch = next(self.retain_iterators[0])
                            retain_inputs = self._prepare_inputs(retain_batch)
                            retain_inputs = {k: v.to(model.device) for k, v in retain_inputs.items()}   
                            
                            prepared_retain_inputs = self._prepare_inputs(retain_inputs)
                            
                            retain_loss_current = self.compute_loss(model, prepared_retain_inputs)
                            
                            with torch.no_grad():
                                retain_loss_oracle = self.compute_loss(frozen_model, prepared_retain_inputs)
                            
                            retain_neg_log_ratios = retain_loss_current - retain_loss_oracle
                            
                            retain_npo_loss = self.args.npo_retain_alpha * (F.logsigmoid(self.args.npo_beta * retain_neg_log_ratios).mean() * 2 / self.args.npo_beta)

                    except StopIteration:
                        # if forget_dataloader has finished, resetting iterators.
                        logger.warning("After resetting, forget_dataloaderis still exhausted. Skipping loss computation.")
                        return torch.tensor(0.0, device=model.device)
                
                # import ipdb; ipdb.set_trace()
                # compute llava default loss
                inputs = self._prepare_inputs(inputs)     
                llava_loss = self.compute_loss(model, inputs)
                npo_loss = forget_npo_loss + retain_npo_loss
                
                loss = npo_loss + self.args.npo_llava_loss_weight * llava_loss
                
                # record loss information to loss.json
                if self.loss_dir:
                    loss_entry = {
                        "step": self.state.global_step,
                        "loss": loss.item(),
                        "llava_loss": llava_loss.item(),
                        "forget_po_loss": forget_npo_loss.item(),
                        "retain_po_loss": retain_npo_loss.item(),
                        "learning_rate": self._get_learning_rate(),
                        "epoch": self.state.epoch
                    }
                    try:
                        with open(self.loss_file_path, 'a') as f:
                            f.write(json.dumps(loss_entry) + '\n\n')  
                    except Exception as e:
                        logger.error(f"Failed to write loss entry to {self.loss_file_path}: {e}")
                        
                if self.args.verbose:
                    print(f"Step {self.state.global_step}: Total Loss={loss}, LLaVA Loss={llava_loss}, Unlearn PO Loss={forget_npo_loss}, Retain PO Loss={retain_npo_loss}")
            
            elif self.args.unlearn_type == "grad-diff":
                pass
            
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                    
            output_loss = loss.detach() / self.args.gradient_accumulation_steps
            return output_loss
        else:
            # using training_step function from the parent Trainer class
            return super().training_step(model, inputs)
                
    def compute_unlearn_loss(self, model: nn.Module, forget_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the unlearning loss.

        Args:
            model (nn.Module): The main training model.
            forget_batch (Dict[str, torch.Tensor]): The current batch of data to be unlearned.

        Returns:
            torch.Tensor: The unlearning loss value.
        """
        updated_forget_activations =  self.forward_with_cache(model, forget_batch, module=self.updated_lora_modules, no_grad=False).to(self.model.device)
        # size:[batch_size, seq_length, hidden_size]
        
        # obtain the size updated_forget_activations
        batch_size, seq_len, _ = updated_forget_activations.shape
        
        # use broadcast_to to expand self.control_vector
        expanded_control_vector = self.control_vector.broadcast_to(batch_size, seq_len, -1)

        # unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, self.control_vector)
        unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, expanded_control_vector)
        
        return unlearn_loss
 
    def test_model(self, model, frozen_model: nn.Module, retain_batch: Dict[str, torch.Tensor]):
        frozen_model.eval()
        prepared_inputs = self._prepare_inputs(retain_batch)
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            try:
                outputs = model(**prepared_inputs)
                print("model forward pass successful.")
            except Exception as e:
                print(f"Error during model forward pass: {e}")
        with torch.no_grad():
            try:
                outputs = frozen_model(**prepared_inputs)
                print("frozen_model forward pass successful.")
            except Exception as e:
                print(f"Error during frozen_model forward pass: {e}")
                
    def compute_retain_loss(self, model: nn.Module, frozen_model: nn.Module, retain_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the retain loss to ensure that the activations of the main model on the retained data remain consistent with those of the frozen_model.

        Args:
            model (nn.Module): The main training model.
            frozen_model (nn.Module): A frozen copy of the model.
            forget_batch (Dict[str, torch.Tensor]): The current batch of data to be unlearned.

        Returns:
            torch.Tensor: The retain loss value.
        """
        updated_retain_activations = self.forward_with_cache(model, retain_batch, module=self.updated_lora_modules, no_grad=False).to(self.model.device)
        updated_forget_activations = self.forward_with_cache(frozen_model, retain_batch, module=self.frozen_lora_modules, no_grad=True).to(self.model.device)
        
        # process batch size and sequence length
        batch_size_retain, seq_len_retain, feature_dim_retain = updated_retain_activations.shape
        batch_size_forget, seq_len_forget, feature_dim_forget = updated_forget_activations.shape
        # print('batch_size_retain:', batch_size_retain, 'seq_len_retain:', seq_len_retain, 'feature_dim_retain:', feature_dim_retain)
        # print('batch_size_forget:', batch_size_forget, 'seq_len_forget:', seq_len_forget, 'feature_dim_forget:', feature_dim_forget)
        
        # make sure batch size is consistent between retain and forget activations
        assert batch_size_retain == batch_size_forget, "batch size is not consistent between retain and forget activations."
        
        min_seq_len = min(seq_len_retain, seq_len_forget)
        updated_retain_activations = updated_retain_activations[:, :min_seq_len, :]
        updated_forget_activations = updated_forget_activations[:, :min_seq_len, :]
        
        retain_loss = torch.nn.functional.mse_loss(updated_retain_activations, updated_forget_activations)
        
        # Assume alpha is a list with the same length as forget_corpora
        alpha = float(self.args.rmu_retain_alpha)  
        return alpha * retain_loss
                            
    def forward_with_cache(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        module: nn.Module, 
        no_grad: bool) -> torch.Tensor:
        
        activations = []

        def hook(module, input, output):
            # print(f"Hook triggered for module: {module}")
            # activations.append(output)
            if isinstance(output, tuple):
                activations.append(output[0])
            else:
                activations.append(output)
            return None 
            
        handle = module.register_forward_hook(hook)
        # # if module is ModuleDict, register hooks for each sub-module 
        # if isinstance(module, nn.ModuleDict):
        #     handles = []
        #     for name, sub_module in module.items():
        #         # print(f"Registering hook on sub-module: {sub_module}")
        #         handles.append(sub_module.register_forward_hook(hook))
        # else:
        #     # print(f"Registering hook on module: {module}")
        #     handle = module.register_forward_hook(hook)

        # use Trainer _prepare_inputs function to prepare inputs
        prepared_inputs = self._prepare_inputs(inputs)

        base_model = get_base_model(model)

        if no_grad:
            with torch.no_grad():
                base_model(**prepared_inputs)
        else:
            base_model(**prepared_inputs)

        # # remove the hook
        # if isinstance(module, nn.ModuleDict):
        #     for handle in handles:
        #         handle.remove()
        # else:
        #     handle.remove()
        handle.remove()

        if len(activations) == 0:
            print(f"No activations captured for module: {module}")
            raise ValueError("No activations were captured. Please check if the hook is correctly registered.")

        return activations[0]
        
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVAUnlearnTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVAUnlearnTrainer, self)._save(output_dir, state_dict)
