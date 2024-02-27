
## Code and Dataset for `Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects`

This repo contains the dataset and model proposed in our IROS 2023 paper:

```
"Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects".
Nikhil Mishra, Pieter Abbeel, Xi Chen, and Maximilan Sieb.
In the proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.
```

The dataset, COB-3D-v2, can be downloaded from our [project page](https://sites.google.com/view/fcon-packing/).

The shape-completion model we proposed, F-CON, is implemented in `fcon_model.py`.

The Jupyter notebook `example.ipynb` shows how to load the data and do inference with a pretrained F-CON model we have provided.


### Citations

If you use COB-3D-v2 or F-CON in your work, please cite:
 ```
 @inproceedings{
     mishra2023convolutional,
     title={Convolutional Occupancy Models for Dense Packing of Complex, Novel Objects},
     author={Nikhil Mishra and Pieter Abbeel and Xi Chen and Maximilian Sieb},
     year={2023},
     booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
}
```

### License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work, including the paper, code, weights, and dataset, is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
