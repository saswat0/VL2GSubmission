## Task 1

The above code rearranges the files appropriately into separate train/val/test directories

### Instructions
* Unzip the dataset and place it in the repository's root directory
    ```bash
    unzip charts.zip -d charts
    ```
* Run the rearrangement file
    ```bash
    cd task1
    python rearrange.py
    ```

### Final directory structure

```
dataset
├── train
│   ├── dot_line
│   ├── hbar_categorical
│   ├── line
│   ├── pie
│   └── vbar_categorical
├── val
│   ├── dot_line
│   ├── hbar_categorical
│   ├── line
│   ├── pie
│   └── vbar_categorical
└── test
```