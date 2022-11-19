cd src/ensembles

python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=123
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=100
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=150
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=0
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=5
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=10
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=20
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=30
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=40
python train_ensembles.py --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed=50
python test_ensembles.py --N=1 --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed='123,100,150,0,5,10,20,30,40,50'
python test_ensembles.py --N=5 --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed='123,100,150,0,5,10,20,30,40,50'
python test_ensembles.py --N=10 --batch_size=16 --lr=0.001 --optimizer=adam --backbone=resnet50 --random_seed='123,100,150,0,5,10,20,30,40,50'
