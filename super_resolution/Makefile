SHELL := /bin/bash # Use bash syntax
PROJECT_NAME=espcn
DOCKER_IMAGE_NAME=${PROJECT_NAME}/pytorch:1.5-cuda10.1-cudnn7-devel

build:
	docker build -t ${DOCKER_IMAGE_NAME} .
run:
	docker run --gpus all -it --user $(shell id -u):$(shell id -g) -v ${HOME}/workspace:/workspace --net host ${DOCKER_IMAGE_NAME} bash

EPOCH=100
train:
	python main.py --cuda --upscale_factor 2 --batchSize 4 --testBatchSize 100 --nEpochs ${EPOCH} --lr 0.001

eval_sr:
	mkdir -p ./isp_test_images/SR_espcn
	i=${EPOCH} ; while [[ $$i -le ${EPOCH} ]] ; do \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/LR_bicubic/Scene4.png' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene4_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/LR_bicubic/Scene5.png' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene5_epoch$$i.png ; \
		((i = i + 1)) ; \
	done
eval_sr_isp:
	mkdir -p ./isp_test_images/SR_espcn
	i=${EPOCH} ; while [[ $$i -le ${EPOCH} ]] ; do \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/ISP_LR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/ISP_LR/Scene4.jpg' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene4_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/ISP_LR/Scene5.jpg' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene5_epoch$$i.png ; \
		((i = i + 1)) ; \
	done
eval_denoise_sr:
	mkdir -p ./isp_test_images/SR_espcn
	python super_resolve.py --cuda \
		--input_image './isp_test_images/LR_bicubic_awgn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' \
		--model model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png
	python super_resolve.py --cuda \
		--input_image ./isp_test_images/LR_bicubic_awgn/Scene4.png \
		--model model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/SR_espcn/Scene4_epoch${EPOCH}.png
	python super_resolve.py --cuda \
		--input_image ./isp_test_images/LR_bicubic_awgn/Scene5.png \
		--model model_epoch_${EPOCH}.pth \
		--output_filename ./isp_test_images/SR_espcn/Scene5_epoch${EPOCH}.png


gen_test_lr_images:
	mkdir -p ./isp_test_images/LR_bilinear
	convert './isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' -filter Triangle -resize 50% './isp_test_images/LR_bilinear/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/HR/Scene4.jpg -filter Triangle -resize 50% ./isp_test_images/LR_bilinear/Scene4.png
	convert ./isp_test_images/HR/Scene5.jpg -filter Triangle -resize 50% ./isp_test_images/LR_bilinear/Scene5.png
	mkdir -p ./isp_test_images/LR_bicubic
	convert './isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' -filter Catrom -resize 50% './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/HR/Scene4.jpg -filter Catrom -resize 50% ./isp_test_images/LR_bicubic/Scene4.png
	convert ./isp_test_images/HR/Scene5.jpg -filter Catrom -resize 50% ./isp_test_images/LR_bicubic/Scene5.png
	mkdir -p ./isp_test_images/SR_bicubic
	convert './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' -filter Catrom -resize 200% './isp_test_images/SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/LR_bicubic/Scene4.png -filter Catrom -resize 200% ./isp_test_images/SR_bicubic/Scene4.png
	convert ./isp_test_images/LR_bicubic/Scene5.png -filter Catrom -resize 200% ./isp_test_images/SR_bicubic/Scene5.png
gen_test_noise_images:
	mkdir -p ./isp_test_images/LR_bicubic_awgn
	convert './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' +noise gaussian -attenuate 1 './isp_test_images/LR_bicubic_awgn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/LR_bicubic/Scene4.png +noise gaussian -attenuate 1 ./isp_test_images/LR_bicubic_awgn/Scene4.png
	convert ./isp_test_images/LR_bicubic/Scene5.png +noise gaussian -attenuate 1 ./isp_test_images/LR_bicubic_awgn/Scene5.png

compare_sr:
	python measure.py --result_image ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene4_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene5_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene5.jpg
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
