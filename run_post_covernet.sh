cd src/post_covernet

python train_post_covernet.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50
python test_post_covernet.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50
