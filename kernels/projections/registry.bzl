"""Generated from registry.toml by //kernels:update_registry_bzl. Do not edit."""

ENTRIES = [
    {"name": "proj_qkv_144_p4", "template": "proj_p4_2tg"},
    {"name": "proj_outproj_144_p4", "template": "proj_p4_2tg"},
    {"name": "proj_ffn1_144_p4", "template": "proj_p4_2tg_silu"},
    {"name": "proj_ffn2_144_p4", "template": "proj_p4_2tg"},
    {"name": "proj_qkv_192_p4", "template": "proj_p4"},
    {"name": "proj_outproj_192_p4", "template": "proj_p4"},
    {"name": "proj_ffn1_192_p4", "template": "proj_p4"},
    {"name": "proj_ffn2_192_p4", "template": "proj_p4"},
    {"name": "proj_qkv_240_p4", "template": "proj_p4"},
    {"name": "proj_outproj_240_p4", "template": "proj_p4"},
    {"name": "proj_ffn1_240_p4", "template": "proj_p4"},
    {"name": "proj_ffn2_240_p4", "template": "proj_p4"},
    {"name": "packed_output_linear_240", "template": "packed_output_linear"},
    {"name": "packed_output_linear_silu_240", "template": "packed_output_linear"},
    {"name": "packed_output_linear_192", "template": "packed_output_linear_p4"},
    {"name": "packed_output_linear_silu_192", "template": "packed_output_linear_p4"},
]
