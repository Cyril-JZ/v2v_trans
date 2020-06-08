#!/bin/bash
python test.py \
  --checkpoint outputs/oilpainting_with_temporal/checkpoints/gen_00550000.pt \
	--a2b 0 \
	--input_folder Video/ \
	--output_folder result/ \
	--num_style 1 \
	--synchronized --seed 10