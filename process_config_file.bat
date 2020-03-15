SET CONFIG_ROOT=wallet_of_torch_renderer/blackbox20_render_configs_cube_slice_8x8/
SET IMG_HEIGHT=24
SET IMG_WIDTH=32

python torch_visualize_config_gen.py %CONFIG_ROOT% %IMG_HEIGHT% %IMG_WIDTH%