#!/bin/bash

python binary_concept_pj.py --seed 0 --n_im 5 --prompt0 "a black dog sitting on the beach" --prompt_plus "a dog"  --prompt_minus "a cat" --prompt_z "a cat"

python binary_concept_pj.py --seed 0 --n_im 5 --prompt0 "a black dog sitting on the beach" --prompt_plus "the beach"  --prompt_minus "the forest" --prompt_z "the forest"

python binary_concept_pj.py --seed 0 --n_im 5 --prompt0 "a black dog sitting on the beach" --prompt_plus "a black dog"  --prompt_minus "a yellow dog" --prompt_z "a yellow dog"

python binary_concept_pj.py --seed 0 --n_im 5 --prompt0 "a boy playing the guitar" --prompt_plus "young man"  --prompt_minus "old man" --prompt_z "an old man"

python binary_concept_pj.py --seed 0 --n_im 5 --prompt0 "a portrait of a man wearing formal clothes" --prompt_plus "formal clothes"  --prompt_minus "casual clothes" --prompt_z "casual clothes"

python binary_concept_pj.py --seed 0 --n_im 5 --prompt0 "people sitting on the grass on a sunny afternoon besides the river" --prompt_plus "a sunny day"  --prompt_minus "a rainy day" --prompt_z "a rainy afternoon"

python binary_concept_pj.py --seed 0 --n_im 5 --prompt0 "a portrait of a smiling woman" --prompt_plus "a happy person"  --prompt_minus "a sad person" --prompt_z "a gloomy woman"




