import copy
import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vime.utils import accelerator

from .actor_group import RayTrainGroup
from .utils import add_default_ray_env_vars

logger = logging.getLogger(__name__)


# @ray.remote(num_gpus=1)
@ray.remote
class InfoActor:
    def get_ip_and_gpu_id(self):
        if accelerator.device_type() == "npu":
            npu_ids = ray.get_runtime_context().get_accelerator_ids().get("NPU", [])
            if npu_ids:
                return ray.util.get_node_ip_address(), npu_ids[0]

        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return ray.util.get_node_ip_address(), gpu_ids[0]

        raise RuntimeError(
            "No GPU/NPU IDs found. "
            f"Accelerator IDs: {ray.get_runtime_context().get_accelerator_ids()}, GPU IDs: {gpu_ids}"
        )


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, int(gpu_id))


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    device_name = accelerator.resource_name()
    bundles = [{device_name: 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    # Wait for the placement group to be scheduled. Poll rather than a bare
    # ray.get(pg.ready()) so the wait is observable: when it can't be placed yet
    # (a node's GPUs haven't registered with the GCS, or an autoscaler is still
    # bringing nodes up) log the GPU counts periodically instead of hanging with no
    # output. The wait stays unbounded, so autoscaling clusters — where a pending
    # placement group is what drives scale-up — are unaffected.
    ready_ref = pg.ready()
    elapsed = 0
    log_interval = 30
    while not ray.wait([ready_ref], timeout=log_interval)[0]:
        elapsed += log_interval
        total = ray.cluster_resources().get(device_name, 0)
        available = ray.available_resources().get(device_name, 0)
        logger.info(
            f"Waiting for placement group of {num_gpus} {device_name}s (elapsed {elapsed}s): "
            f"{total:g} {device_name}s registered with Ray, {available:g} available."
        )

    # use info actor to get the GPU id
    info_actors = []
    info_actor_resources = {"num_cpus": 1}
    if device_name == "GPU":
        info_actor_resources["num_gpus"] = 1
    else:
        info_actor_resources["resources"] = {device_name: 1}
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                **info_actor_resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0

    if args.rollout_external:
        if args.debug_rollout_only:
            return actor_num_gpus, 0
        return actor_num_gpus, actor_num_gpus

    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0

    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0

    return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result["critic"] = result["actor"] if args.use_critic else None

    return result


def allocate_train_group(
    args,
    num_nodes,
    num_gpus_per_node,
    pg,
    role="actor",
    with_ref=False,
    with_opd_teacher=False,
    actor_cls=None,
):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
        with_ref=with_ref,
        with_opd_teacher=with_opd_teacher,
        actor_cls=actor_cls,
    )


def create_actor_model(args, pgs, rollout_manager, actor_cls=None):
    actor_args = args
    if args.megatron_config_path is not None:
        from vime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model_kwargs = {}
    if actor_cls is not None:
        actor_model_kwargs["actor_cls"] = actor_cls
    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
        with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        **actor_model_kwargs,
    )
    actor_start_rollout_ids = actor_model.create(rollout_manager=rollout_manager)
    return actor_model, actor_start_rollout_ids


def create_training_models(args, pgs, rollout_manager, actor_cls=None):
    actor_model, actor_start_rollout_ids = create_actor_model(
        args,
        pgs,
        rollout_manager,
        actor_cls=actor_cls,
    )

    critic_model = None
    if args.use_critic and args.num_rollout != 0:
        from vime.utils.arguments import parse_megatron_role_args

        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else copy.deepcopy(args)
        )
        if args.megatron_config_path is None:
            critic_args.disable_param_buffers_cpu_backup = False

        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_start_rollout_ids = critic_model.create(rollout_manager=rollout_manager)

    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if critic_model is not None:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if rollout_manager is not None:
        restore_rollout_dataset(args, rollout_manager)

    return actor_model, critic_model


def attach_rollout_manager(args, actor_model, critic_model, rollout_manager) -> None:
    """Attach the ready rollout manager after concurrent actor initialization."""
    actor_model.set_rollout_manager(rollout_manager)
    if critic_model is not None:
        critic_model.set_rollout_manager(rollout_manager)
    restore_rollout_dataset(args, rollout_manager)


def restore_rollout_dataset(args, rollout_manager) -> None:
    """Resume the global dataset cursor after training determines the start ID."""
    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))


def initialize_rollout_and_training(args, pgs):
    """Initialize rollout and training sides, overlapping startup when safe.

    Inferring ``num_rollout`` from the rollout manager requires a blocking Ray
    call, so that case cannot overlap training initialization. In parallel mode,
    rollout failures may surface only after the blocking training init finishes;
    this helper does not provide fail-fast monitoring or cross-side cleanup.
    """
    if getattr(args, "colocate", False):
        serial_reason = "colocate"
    elif getattr(args, "rollout_external", False):
        serial_reason = "rollout_external"
    elif getattr(args, "debug_train_only", False):
        serial_reason = "debug_train_only"
    elif getattr(args, "debug_rollout_only", False):
        serial_reason = "debug_rollout_only"
    elif args.num_rollout is None:
        serial_reason = "num_rollout is unset; inferring it waits for rollout initialization"
    else:
        serial_reason = None
    parallel_startup = serial_reason is None
    logger.info(
        "Starting rollout and training initialization in %s mode%s.",
        "parallel" if parallel_startup else "serial",
        "" if parallel_startup else f" ({serial_reason})",
    )
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args,
        pgs["rollout"],
    )

    if parallel_startup:
        actor_model, critic_model = create_training_models(
            args,
            pgs,
            None,
        )
        router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
        prepare_rollout_manager(args, rollout_manager)
        attach_rollout_manager(args, actor_model, critic_model, rollout_manager)
    else:
        prepare_rollout_manager(args, rollout_manager)
        router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
        actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    return rollout_manager, num_rollout_per_epoch, actor_model, critic_model, router_addr


def create_rollout_manager(args, pg):
    from .rollout import RolloutManager

    device_name = accelerator.resource_name()
    rollout_manager_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "resources": {device_name: 0},
        "runtime_env": {"env_vars": add_default_ray_env_vars()},
    }
    if getattr(args, "rollout_data_transport", "object-store") == "nixl":
        rollout_manager_options["enable_tensor_transport"] = True
    rollout_manager = RolloutManager.options(**rollout_manager_options).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    return rollout_manager, num_rollout_per_epoch


def prepare_rollout_manager(args, rollout_manager) -> None:
    """Run rollout setup that must happen after the manager is initialized."""
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
