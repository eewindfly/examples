SHELL := /bin/bash # Use bash syntax
PROJECT_NAME=espcn
DOCKER_IMAGE_NAME=${PROJECT_NAME}/pytorch:1.5-cuda10.1-cudnn7-devel


build:
	docker build -t ${DOCKER_IMAGE_NAME} .
run:
	docker run \
		--gpus all \
		-it \
		--user $(shell id -u):$(shell id -g) \
		--env="DISPLAY" \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ${HOME}/workspace:/workspace \
		--net host \
		--shm-size 8G \
		${DOCKER_IMAGE_NAME} bash


tb:
	tensorboard --logdir ./logs


EPOCH=100
train_sr:
	python train_sr.py --upscale_factor 2 --batchSize 4 --nEpochs ${EPOCH} --lr 0.001 --residual
train_denoising:
	python train_denoising.py --batchSize 4 --nEpochs ${EPOCH} --lr 0.001
train_denoising_sr:
	python train_denoising_sr.py --upscale_factor 2 --batchSize 4 --nEpochs ${EPOCH} --lr 0.001


eval_sr:
	python eval.py \
		--input_dir ./isp_test_images/LR_bicubic \
		--output_dir ./isp_test_images/SR_espcn \
		--model_dir ./models/sr
eval_sr_isp:
	mkdir -p ./isp_test_images/SR_espcn
	i=${EPOCH} ; while [[ $$i -le ${EPOCH} ]] ; do \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/ISP_LR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' \
			--model ./sr_models/model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/ISP_LR/Scene4.jpg' \
			--model ./sr_models/model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene4_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/ISP_LR/Scene5.jpg' \
			--model ./sr_models/model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene5_epoch$$i.png ; \
		((i = i + 1)) ; \
	done
eval_denoising:
	mkdir -p ./isp_test_images/DN_ours
	python test_denoising.py --cuda \
		--input_image './isp_test_images/LR_bicubic_awgn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' \
		--model ./denoising_models/model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/DN_ours/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png
	python test_denoising.py --cuda \
		--input_image ./isp_test_images/LR_bicubic_awgn/Scene4.png \
		--model ./denoising_models/model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/DN_ours/Scene4_epoch${EPOCH}.png
	python test_denoising.py --cuda \
		--input_image ./isp_test_images/LR_bicubic_awgn/Scene5.png \
		--model ./denoising_models/model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/DN_ours/Scene5_epoch${EPOCH}.png
eval_denoising_sr:
	mkdir -p ./isp_test_images/DN_SR_espcn
	python super_resolve.py --cuda \
		--input_image './isp_test_images/LR_bicubic_awgn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' \
		--model ./denoising_sr_models/model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/DN_SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png
	python super_resolve.py --cuda \
		--input_image ./isp_test_images/LR_bicubic_awgn/Scene4.png \
		--model ./denoising_sr_models/model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/DN_SR_espcn/Scene4_epoch${EPOCH}.png
	python super_resolve.py --cuda \
		--input_image ./isp_test_images/LR_bicubic_awgn/Scene5.png \
		--model ./denoising_sr_models/model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/DN_SR_espcn/Scene5_epoch${EPOCH}.png


gen_test_lr_images:
	# mkdir -p ./isp_test_images/LR_bilinear
	# convert './isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' -filter Triangle -resize 50% './isp_test_images/LR_bilinear/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	# convert ./isp_test_images/HR/Scene4.jpg -filter Triangle -resize 50% ./isp_test_images/LR_bilinear/Scene4.png
	# convert ./isp_test_images/HR/Scene5.jpg -filter Triangle -resize 50% ./isp_test_images/LR_bilinear/Scene5.png
	mkdir -p ./isp_test_images/LR_bicubic
	convert './isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' -filter Catrom -resize 50% './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/HR/Scene4.jpg -filter Catrom -resize 50% ./isp_test_images/LR_bicubic/Scene4.png
	convert ./isp_test_images/HR/Scene5.jpg -filter Catrom -resize 50% ./isp_test_images/LR_bicubic/Scene5.png
	mkdir -p ./isp_test_images/SR_bicubic
	convert './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' -filter Catrom -resize 200% './isp_test_images/SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/LR_bicubic/Scene4.png -filter Catrom -resize 200% ./isp_test_images/SR_bicubic/Scene4.png
	convert ./isp_test_images/LR_bicubic/Scene5.png -filter Catrom -resize 200% ./isp_test_images/SR_bicubic/Scene5.png
gen_test_noising_images:
	mkdir -p ./isp_test_images/LR_bicubic_awgn
	python3 add_noise.py --sigma 10 --luma_only --input_image './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' --output_filename './isp_test_images/LR_bicubic_awgn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	python3 add_noise.py --sigma 10 --luma_only --input_image ./isp_test_images/LR_bicubic/Scene4.png --output_filename ./isp_test_images/LR_bicubic_awgn/Scene4.png
	python3 add_noise.py --sigma 10 --luma_only --input_image ./isp_test_images/LR_bicubic/Scene5.png --output_filename ./isp_test_images/LR_bicubic_awgn/Scene5.png
	mkdir -p ./isp_test_images/DN_SR_bicubic
	convert './isp_test_images/LR_bicubic_awgn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' -filter Catrom -resize 200% './isp_test_images/DN_SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/LR_bicubic_awgn/Scene4.png -filter Catrom -resize 200% ./isp_test_images/DN_SR_bicubic/Scene4.png
	convert ./isp_test_images/LR_bicubic_awgn/Scene5.png -filter Catrom -resize 200% ./isp_test_images/DN_SR_bicubic/Scene5.png


compare_sr:
	python measure.py --result_image ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene4_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene5_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene5.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/Scene4.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/Scene5.png --gt_image ./isp_test_images/HR/Scene5.jpg
compare_denoising_sr:
	python measure.py --result_image ./isp_test_images/DN_SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/DN_SR_espcn/Scene4_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/DN_SR_espcn/Scene5_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene5.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/Scene4.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/Scene5.png --gt_image ./isp_test_images/HR/Scene5.jpg
compare_sr_isp:
	python measure.py --result_image ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene4_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene5_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene5.jpg
	python measure.py --result_image ./isp_test_images/ISP_SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/ISP_SR_bicubic/Scene4.jpg --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/ISP_SR_bicubic/Scene5.jpg --gt_image ./isp_test_images/HR/Scene5.jpg
