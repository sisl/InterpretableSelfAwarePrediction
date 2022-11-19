cd src/isap

python train_isap.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet18
python test_isap.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet18
