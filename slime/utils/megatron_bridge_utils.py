from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def patch_hf_config_for_megatron_bridge(hf_config):
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    return hf_config


def _patch_model_bridge_for_hf_layout(model_bridge, *, individual_experts):
    if getattr(model_bridge, "_slime_hf_layout_patched", False):
        return model_bridge

    if individual_experts:
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping

        original_mapping_registry = model_bridge.mapping_registry
        expert_fc1 = "language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*"
        expert_fc2 = "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*"

        def mapping_registry_with_individual_experts():
            registry = original_mapping_registry()
            mappings = [
                mapping
                for mapping in registry.mappings
                if mapping.megatron_param not in (expert_fc1, expert_fc2)
            ]
            mappings.extend(
                [
                    GatedMLPMapping(
                        megatron_param=expert_fc1,
                        gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                        up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=expert_fc2,
                        hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )
            return MegatronMappingRegistry(*mappings)

        model_bridge.mapping_registry = mapping_registry_with_individual_experts

    # Megatron Bridge can leave None placeholders in conversion tasks for
    # destination parameters that have no mapping. Its HF loader dereferences
    # every entry, so language-only exports of multimodal models otherwise fail
    # before the mapped language weights can be loaded.
    build_conversion_tasks = model_bridge.build_conversion_tasks

    def build_conversion_tasks_without_missing_entries(*args, **kwargs):
        tasks = build_conversion_tasks(*args, **kwargs)
        return [task for task in tasks if task is not None]

    model_bridge.build_conversion_tasks = build_conversion_tasks_without_missing_entries
    model_bridge._slime_hf_layout_patched = True
    return model_bridge


def patch_auto_bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)

    state = getattr(hf_pretrained, "state", None)
    source = getattr(state, "source", None)
    hf_keys = set(source.get_all_keys()) if source is not None else set()
    individual_expert_key = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    fused_expert_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    individual_experts = individual_expert_key in hf_keys and fused_expert_key not in hf_keys

    hf_config = getattr(hf_pretrained, "config", None)
    if hf_config is not None:
        hf_config._slime_individual_expert_weights = individual_experts

    # AutoBridge._model_bridge is a property that creates a new bridge on every
    # access. Patch the dispatcher so load, export, and weight sync all receive
    # the checkpoint-layout-aware mapping rather than only the first instance.
    from megatron.bridge.models.conversion import model_bridge as model_bridge_module

    get_model_bridge = model_bridge_module.get_model_bridge
    if not getattr(get_model_bridge, "_slime_hf_layout_dispatch", False):

        def get_model_bridge_with_hf_layout(hf_architecture, hf_config=None):
            resolved_bridge = get_model_bridge(hf_architecture, hf_config=hf_config)
            if getattr(hf_config, "_slime_individual_expert_weights", False):
                return _patch_model_bridge_for_hf_layout(resolved_bridge, individual_experts=True)
            return resolved_bridge

        get_model_bridge_with_hf_layout._slime_hf_layout_dispatch = True
        model_bridge_module.get_model_bridge = get_model_bridge_with_hf_layout

    model_bridge = getattr(bridge, "_model_bridge", None)
    if model_bridge is not None and individual_experts:
        _patch_model_bridge_for_hf_layout(model_bridge, individual_experts=True)

    return bridge


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
