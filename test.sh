#!/bin/sh
PARTITION=Segmentation


# sleep 2h
GPU_ID=2
dataset=iSAID # iSAID iSAID_1
arch=R2Net

s_q=False
variable1=
variable2=

for cross_domain in iSAID
do
        for net in vgg resnet50   # vgg resnet50
        do
                for shot in 1 5 # 1 5 
                do
                        for split in 0 1 2 # 0 1 2
                        do
                                exp_dir=exp/${arch}/${dataset}/${net}/split${split} # 
                                snapshot_dir=${exp_dir}/${shot}shot
                                result_dir=${exp_dir}/result
                                mkdir -p ${snapshot_dir} ${result_dir}
                                now=$(date +"%Y%m%d_%H%M%S")

                                echo ${arch}_${dataset}
                                echo ${net}_split${split}_${shot}shot

                                CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
                                                        --arch=${arch} \
                                                        --shot=${shot} \
                                                        --split=${split} \
                                                        --backbone=${net} \
                                                        --dataset=${dataset} \
                                                        --cross_domain=${cross_domain} \
                                                        --s_q=${s_q} \
                                                        --variable1=${variable1} \
                                                        --variable2=${variable2} \
                                                        2>&1 | tee ${result_dir}/test-$now.log        
                        done
                done
        done
done