SET CONFIG_ROOT=wallet_of_torch_renderer/blackbox20_render_configs_1x1/
SET IMG_HEIGHT=192
SET IMG_WIDTH=256

python torch_visualize_config_gen.py %CONFIG_ROOT% %IMG_HEIGHT% %IMG_WIDTH%