docker_start:
	docker build --platform linux/arm64 -t spot-rna-1d-test . 
	docker run -it -v ${PWD}/helper_cli.py:/app/helper_cli.py -v ${PWD}/outputs/:/app/outputs -v ${PWD}/checkpoints:/app/checkpoints -v ${PWD}/data:/app/data spot-rna-1d-test
