eval "$(conda shell.bash hook)"
conda activate sam3
python run_image_mask.py --image_path /home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/MC1/inpaint/0000/image/0000.png \
                            --out_path output/MC1/0000_mask.png