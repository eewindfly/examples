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

test:
	mkdir -p ./isp_test_images/SR_espcn
	i=${EPOCH} ; while [[ $$i -le ${EPOCH} ]] ; do \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' \
			--gt_image './isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/LR_bicubic/Scene4.png' \
			--gt_image './isp_test_images/HR/Scene4.jpg' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene4_epoch$$i.png ; \
		python super_resolve.py --cuda \
			--input_image './isp_test_images/LR_bicubic/Scene5.png' \
			--gt_image './isp_test_images/HR/Scene5.jpg' \
			--model model_epoch_$$i.pth \
			--output_filename ./isp_test_images/SR_espcn/Scene5_epoch$$i.png ; \
		((i = i + 1)) ; \
	done

gen_test_images:
	mkdir -p ./isp_test_images/LR_bicubic
	convert './isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg' -interpolate bicubic -resize 960 './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/HR/Scene4.jpg -interpolate bicubic -resize 960 ./isp_test_images/LR_bicubic/Scene4.png
	convert ./isp_test_images/HR/Scene5.jpg -interpolate bicubic -resize 960 ./isp_test_images/LR_bicubic/Scene5.png
	mkdir -p ./isp_test_images/SR_bicubic
	convert './isp_test_images/LR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png' -interpolate bicubic -resize 1920 './isp_test_images/SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png'
	convert ./isp_test_images/LR_bicubic/Scene4.png -interpolate bicubic -resize 1920 ./isp_test_images/SR_bicubic/Scene4.png
	convert ./isp_test_images/LR_bicubic/Scene5.png -interpolate bicubic -resize 1920 ./isp_test_images/SR_bicubic/Scene5.png

compare:
	python measure.py --result_image ./isp_test_images/SR_espcn/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene4_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/SR_espcn/Scene5_epoch${EPOCH}.png --gt_image ./isp_test_images/HR/Scene5.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.png --gt_image ./isp_test_images/HR/1920X1080_-color=3_-bits=12_-frame=0_ISO12233_iso100.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/Scene4.png --gt_image ./isp_test_images/HR/Scene4.jpg
	python measure.py --result_image ./isp_test_images/SR_bicubic/Scene5.png --gt_image ./isp_test_images/HR/Scene5.jpg
