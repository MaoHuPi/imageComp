# imageComp｜圖片比較

> 2024 &copy; MaoHuPi
> 
> Program Function:
> > Image Similarity Comparison
> 
> Source of Inspiration:
> > [cyris - python 圖像比較一問](https://ithelp.ithome.com.tw/questions/10215126)

## Description

I start from three simpler aspects, namely:

1. Color comparison
2. Image features
3. Ease of confusion

## Arguments

* `-h`, `--help` (bool)  
	Default value: False.  
	Show this arguments description.
* `-v`, `--version` (bool)  
	Default value: False.  
	Show the current version.
* `-s`, `--source_dir` (str)  
	Set the directory of source images. Selected image must in this directory.
* `-t`, `--target_dir` (str|False)  
	Default value: False.  
	Set the directory to which the output file will be copy.
* `-i`, `--selected_image` (str)  
	Set the image that the other images should be compare with.
* `-w`, `--hsv_w` (list[3])  
	Default value: [1, 0.2, 0.2].  
	Set the degree of influence of h, s and v on the compare score which used for comparison.
* `-u`, `--update_catch` (bool)  
	Default value: True.  
	Update catch data after the weight fitted.
* `-c`, `--use_catch` (bool)  
	Default value: True.  
	Use catch data when it is exist.
* `-m`, `--compare_mode` (str)  
	Default value: head.  
	Can be head|last|all.
* `-l`, `--output_number_limit` (int)  
	Default value: 5.  
	Set the maxima output number.
* `-a`, `--min_val_accuracy` (float)  
	Default value: 0.7.  
	Set the 'val_accuracy' value to early stopping the training process.

## Usage

```cmd
python main.py -s "image/source_cat" -t "image/target" -i "cat (34).jpg" -m head -l 5
```