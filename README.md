# FARSA: Fully Automated Roadway Safety Assessment
This repository contains the model files, solver definitions, and
learned weights for the networks described in the following
publication:

> FARSA: Fully Automated Roadway Safety Assessment (Weilian Song, Scott Workman, Armin Hadzic, Xu Zhang, Eric Green, Mei Chen, Reginald Souleyrette, Nathan Jacobs),
> In IEEE Winter Conference on Applications of Computer Vision (WACV), 2018.
```
@inproceedings{song2018farsa,
  author={Song, Weilian and Workman, Scott and Hadzic, Armin and Zhang, Xu and Green, Eric and Chen, Mei and Souleyrette, Reginald and Jacobs, Nathan},
  title={FARSA: Fully Automated Roadway Safety Assessment},
  booktitle={{IEEE Winter Conference on Applications of Computer Vision (WACV)}},
  year={2018}
}
```
## Getting Started
Download the trained weights required for the inference [here](), and put it inside a 
folder named `checkpoint/` at the root of the repo.

Run the inference script by typing `python infer.py`, or pass in your own panorama image this way:

```python infer.py --img_path path_to_image```

## Note
Labels for three extra  auxiliary tasks were used during training
(presence of shoulder rumble strips, center rumble strips, and motorcycle facilities),
however the label distributions for these
three tasks are extremely skewed, therefore they are ignored in the paper.

## License
This software is released under a creative commons license which 
allows for personal and research use only. For a commercial license
please contact the authors. You can view a license summary here:
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

## Contact
Weilian Song  
University of Kentucky  
http://cs.uky.edu/~wso226/
