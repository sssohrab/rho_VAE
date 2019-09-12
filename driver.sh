#
python main.py --uid 'VanillaVAE' --epochs 1 --dataset-name 'mnist'
python main.py --uid 'INFO_VAE' --epochs 1 --dataset-name 'mnist'
python main.py --uid 'CNN_VAE' --epochs 1 --dataset-name 'mnist'
python main.py --uid 'BETAVAE' --epochs 1 --dataset-name 'mnist'


python main.py --uid 'RHO_VanillaVAE' --epochs 1 --dataset-name 'mnist' --rho
python main.py --uid 'RHO_INFO_VAE' --epochs 1 --dataset-name 'mnist' --rho
python main.py --uid 'RHO_CNN_VAE' --epochs 1 --dataset-name 'mnist' --rho
python main.py --uid 'RHO_BETAVAE' --epochs 1 --dataset-name 'mnist' --rho


python main.py --uid 'VanillaVAE' --epochs 1 --dataset-name 'fashion'
python main.py --uid 'INFO_VAE' --epochs 1 --dataset-name 'fashion'
python main.py --uid 'CNN_VAE' --epochs 1 --dataset-name 'fashion'
python main.py --uid 'BETAVAE' --epochs 1 --dataset-name 'fashion'


python main.py --uid 'RHO_VanillaVAE' --epochs 1 --dataset-name 'fashion' --rho
python main.py --uid 'RHO_INFO_VAE' --epochs 1 --dataset-name 'fashion' --rho
python main.py --uid 'RHO_CNN_VAE' --epochs 1 --dataset-name 'fashion' --rho
python main.py --uid 'RHO_BETAVAE' --epochs 1 --dataset-name 'fashion' --rho