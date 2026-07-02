# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import types
import json
from typing import Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from packaging import version as pkg_version

try:
    import triton  # noqa: F401 # type: ignore
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from . import ops
from . import module_inject

from .accelerator import get_accelerator
from .runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
from .runtime.engine import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from .runtime.hybrid_engine import DeepSpeedHybridEngine
from .runtime.pipe.engine import PipelineEngine
from .inference.engine import InferenceEngine
from .inference.config import DeepSpeedInferenceConfig
from .runtime.lr_schedules import add_tuning_arguments
from .runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from .runtime.activation_checkpointing import checkpointing
from .ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from .module_inject import replace_transformer_layer, revert_transformer_layer

from .utils import log_dist, OnDevice, logger
from .comm.comm import init_distributed

from .runtime import zero
from .runtime import DeepSpeedOptimizer, ZeROOptimizer

from .pipe import PipelineModule

from .git_version_info import version, git_hash, git_branch


def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro


# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(
    __version__)
__git_hash__ = git_hash
__git_branch__ = git_branch

# Set to torch's distributed package or deepspeed.comm based inside DeepSpeedEngine init
dist = None


class CheckFreqPipelineEngine(PipelineEngine):
    def __init__(self, has_bool_tensors=False, checkp_params={}, *super_args, **super_kwargs):
        super().__init__(has_bool_tensors, *super_args, **super_kwargs)
        self.in_progress_snapshot = checkp_params['in_progress_snapshot']
        print("Inside CheckFreqPipelineEngine")

    def _exec_optimizer_step(self, lr_kwargs=None):
        while self.in_progress_snapshot.value == 1:
            continue
        super()._exec_optimizer_step(lr_kwargs)


class PCCheckPipelineEngine(PipelineEngine):
    def __init__(self, has_bool_tensors=False, checkp_params={}, *super_args, **super_kwargs):
        super().__init__(has_bool_tensors, *super_args, **super_kwargs)
        self.chk_monitor = None

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.chk_monitor is not None:
            while self.chk_monitor.gpu_copy_in_progress():
                continue
        super()._exec_optimizer_step(lr_kwargs)


class MultiStreamPipelineEngine(PipelineEngine):
    """集成 multistream 异步检查点的 DeepSpeed Pipeline Engine。

    每个 PP 阶段独立使用 MultiStreamCheckpoint 保存自己的参数和优化器状态，
    恢复时通过 pipelayer 的 MultiStreamStateLoader 按阶段加载。
    """

    def __init__(self, has_bool_tensors=False, checkp_params=None,
                 *super_args, **super_kwargs):
        super().__init__(has_bool_tensors, *super_args, **super_kwargs)
        checkp_params = checkp_params or {}
        self._ms_checkpoint_dir = checkp_params.get(
            "checkpoint_dir", "./ms_checkpoints")
        self._ms_lib_path = checkp_params.get(
            "lib_path", "./libtest_ssd.so")
        self._ms_max_async = int(checkp_params.get("max_async", 2))
        self._ms_num_layer_groups = int(
            checkp_params.get("num_layer_groups", 8))
        self._ms_num_threads = int(checkp_params.get("num_threads", 16))

        self._ms_ckpt = None
        self._ms_gpu_buffer = None
        self._ms_param_layout = None
        self._ms_initialized = False

    def _init_multistream(self):
        if self._ms_initialized:
            return

        import os
        import sys
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from checkpoint_eval.pccheck.multistream_pipeline_utils import (
            create_stage_checkpoint,
        )

        self._ms_ckpt, self._ms_gpu_buffer, self._ms_param_layout = (
            create_stage_checkpoint(
                model=self.module,
                optimizer=self.optimizer,
                checkpoint_dir=self._ms_checkpoint_dir,
                lib_path=self._ms_lib_path,
                rank=self.global_rank,
                world_size=(self.grid.data_parallel_size
                            * self.grid.pipe_parallel_size),
                max_async=self._ms_max_async,
                num_layer_groups=self._ms_num_layer_groups,
                num_threads=self._ms_num_threads,
            )
        )
        self._ms_initialized = True
        if self.global_rank == 0:
            print("[MultiStreamPipelineEngine] Checkpoint initialized "
                  f"for rank {self.global_rank}")

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self._ms_ckpt is not None:
            for events in self._ms_ckpt._layer_copy_events.values():
                if events and events[0] is not None:
                    torch.cuda.current_stream().wait_event(events[0])
        super()._exec_optimizer_step(lr_kwargs)

    def save_multistream_checkpoint(self, step, sync=False, return_breakdown=False):
        """触发 multistream 异步保存，默认返回耗时（秒）。"""
        if not self._ms_initialized:
            self._init_multistream()

        import os
        import sys
        import time
        repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from checkpoint_eval.pccheck.multistream_pipeline_utils import (
            save_stage_checkpoint,
            save_global_training_state,
        )

        stage_result = save_stage_checkpoint(
            ms_ckpt=self._ms_ckpt,
            gpu_buffer=self._ms_gpu_buffer,
            param_layout=self._ms_param_layout,
            model=self.module,
            optimizer=self.optimizer,
            checkpoint_dir=self._ms_checkpoint_dir,
            rank=self.global_rank,
            step=step,
            sync=sync,
            return_breakdown=return_breakdown,
        )

        if return_breakdown:
            breakdown = dict(stage_result)
            elapsed = float(breakdown.get("elapsed_sec", 0.0))
        else:
            elapsed = stage_result

        global_state_ms = 0.0
        if self.global_rank == 0:
            global_state_t0 = time.perf_counter()
            save_global_training_state(
                self._ms_checkpoint_dir, step)
            global_state_ms = (time.perf_counter() - global_state_t0) * 1000.0

        barrier_ms = 0.0
        if torch.distributed.is_initialized():
            barrier_t0 = time.perf_counter()
            torch.distributed.barrier()
            barrier_ms = (time.perf_counter() - barrier_t0) * 1000.0

        if return_breakdown:
            metadata_barrier_ms = (
                float(breakdown.get("metadata_ms", 0.0))
                + float(breakdown.get("stage_state_ms", 0.0))
                + global_state_ms
                + barrier_ms
            )
            total_ms = float(breakdown.get("total_ms", elapsed * 1000.0))
            total_ms += global_state_ms + barrier_ms
            breakdown.update({
                "elapsed_sec": total_ms / 1000.0,
                "total_ms": total_ms,
                "global_state_ms": global_state_ms,
                "barrier_ms": barrier_ms,
                "metadata_barrier_ms": metadata_barrier_ms,
            })
            return breakdown

        return elapsed

    def shutdown_multistream(self):
        if self._ms_ckpt is not None:
            self._ms_ckpt.shutdown()
            self._ms_ckpt = None


def initialize(args=None,
               model: torch.nn.Module = None,
               optimizer: Optional[Union[Optimizer,
                                         DeepSpeedOptimizerCallable]] = None,
               model_parameters: Optional[torch.nn.Module] = None,
               training_data: Optional[torch.utils.data.Dataset] = None,
               lr_scheduler: Optional[Union[_LRScheduler,
                                            DeepSpeedSchedulerCallable]] = None,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               collate_fn=None,
               config=None,
               config_params=None,
               checkp_type=None,
               checkp_params=None):
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(__version__, __git_hash__,
                                                                             __git_branch__),
             ranks=[0])

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, "deepspeed.initialize requires a model"

    global dist
    from deepspeed import comm as dist
    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend,
                          dist_init_required=dist_init_required)

    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    # Check for deepscale_config for backwards compat
    if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
        logger.warning(
            "************ --deepscale_config is deprecated, please use --deepspeed_config ************")
        if hasattr(args, "deepspeed_config"):
            assert (args.deepspeed_config is
                    None), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        assert config is None, "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
        config = args.deepspeed_config
    assert config is not None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    if not isinstance(model, PipelineModule):
        config_class = DeepSpeedConfig(config, mpu)
        if config_class.hybrid_engine.enabled:
            engine = DeepSpeedHybridEngine(args=args,
                                           model=model,
                                           optimizer=optimizer,
                                           model_parameters=model_parameters,
                                           training_data=training_data,
                                           lr_scheduler=lr_scheduler,
                                           mpu=mpu,
                                           dist_init_required=dist_init_required,
                                           collate_fn=collate_fn,
                                           config=config,
                                           config_class=config_class)
        else:
            engine = DeepSpeedEngine(args=args,
                                     model=model,
                                     optimizer=optimizer,
                                     model_parameters=model_parameters,
                                     training_data=training_data,
                                     lr_scheduler=lr_scheduler,
                                     mpu=mpu,
                                     dist_init_required=dist_init_required,
                                     collate_fn=collate_fn,
                                     config=config,
                                     config_class=config_class)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        mpu = model.mpu()
        config_class = DeepSpeedConfig(config, mpu)

        print(f'-------------------- checkp_type is {checkp_type}')
        checkp_type_norm = (
            checkp_type.strip().lower()
            if isinstance(checkp_type, str)
            else ""
        )

        if checkp_type_norm == 'checkfreq':
            engine = CheckFreqPipelineEngine(
                checkp_params=checkp_params,
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                config_class=config_class)
        elif checkp_type_norm == 'pccheck':
            engine = PCCheckPipelineEngine(
                checkp_params=checkp_params,
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                config_class=config_class)
        elif checkp_type_norm == 'multistream':
            engine = MultiStreamPipelineEngine(
                checkp_params=checkp_params,
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                config_class=config_class)
        else:
            engine = PipelineEngine(args=args,
                                    model=model,
                                    optimizer=optimizer,
                                    model_parameters=model_parameters,
                                    training_data=training_data,
                                    lr_scheduler=lr_scheduler,
                                    mpu=mpu,
                                    dist_init_required=dist_init_required,
                                    collate_fn=collate_fn,
                                    config=config,
                                    config_class=config_class)

    # Restore zero.Init context if necessary
    zero.partition_parameters.restore_init_context()

    return_items = [engine, engine.optimizer,
                    engine.training_dataloader, engine.lr_scheduler]
    return tuple(return_items)


def _add_core_arguments(parser):
    r"""Helper (internal) function to update an argument parser with an argument group of the core DeepSpeed arguments.
        The core set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

        This is a helper function to the public add_config_arguments()

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')

    group.add_argument('--deepspeed',
                       default=False,
                       action='store_true',
                       help='Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepspeed_config', default=None,
                       type=str, help='DeepSpeed json configuration file.')

    group.add_argument('--deepscale',
                       default=False,
                       action='store_true',
                       help='Deprecated enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepscale_config',
                       default=None,
                       type=str,
                       help='Deprecated DeepSpeed json configuration file.')

    group.add_argument('--deepspeed_mpi',
                       default=False,
                       action='store_true',
                       help="Run via MPI, this will attempt to discover the necessary variables to initialize torch "
                       "distributed from the MPI environment")

    return parser


def add_config_arguments(parser):
    r"""Update the argument parser to enabling parsing of DeepSpeed command line arguments.
        The set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    parser = _add_core_arguments(parser)

    return parser


def default_inference_config():
    """
        Return a default DeepSpeed inference configuration dictionary.
    """
    return DeepSpeedInferenceConfig().dict()


def init_inference(model, config=None, **kwargs):
    """Initialize the DeepSpeed InferenceEngine.

    Description: all four cases are valid and supported in DS init_inference() API.

    # Case 1: user provides no config and no kwargs. Default config will be used.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model)
        string = generator("DeepSpeed is")
        print(string)

    # Case 2: user provides a config and no kwargs. User supplied config will be used.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model, config=config)
        string = generator("DeepSpeed is")
        print(string)

    # Case 3: user provides no config and uses keyword arguments (kwargs) only.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model,
                                                    mp_size=world_size,
                                                    dtype=torch.half,
                                                    replace_with_kernel_inject=True)
        string = generator("DeepSpeed is")
        print(string)

    # Case 4: user provides config and keyword arguments (kwargs). Both config and kwargs are merged and kwargs take precedence.

    .. code-block:: python

        generator.model = deepspeed.init_inference(generator.model, config={"dtype": torch.half}, replace_with_kernel_inject=True)
        string = generator("DeepSpeed is")
        print(string)

    Arguments:
        model: Required: original nn.module object without any wrappers

        config: Optional: instead of arguments, you can pass in a DS inference config dict or path to JSON file

    Returns:
        A deepspeed.InferenceEngine wrapped model.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(__version__, __git_hash__,
                                                                             __git_branch__),
             ranks=[0])

    # Load config_dict from config first
    if config is None:
        config = {}
    if isinstance(config, str):
        with open(config, "r") as f:
            config_dict = json.load(f)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(
            f"'config' argument expected string or dictionary, got {type(config)}")

    # Update with values from kwargs, ensuring no conflicting overlap between config and kwargs
    overlap_keys = set(config_dict.keys()).intersection(kwargs.keys())
    # If there is overlap, error out if values are different
    for key in overlap_keys:
        if config_dict[key] != kwargs[key]:
            raise ValueError(
                f"Conflicting argument '{key}' in 'config':{config_dict[key]} and kwargs:{kwargs[key]}")
    config_dict.update(kwargs)

    ds_inference_config = DeepSpeedInferenceConfig(**config_dict)

    engine = InferenceEngine(model, config=ds_inference_config)

    return engine
