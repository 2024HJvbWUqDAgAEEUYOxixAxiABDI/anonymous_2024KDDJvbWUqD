## Code for Hyperbolic Transformer

- The dataset will be downloaded automatically but you may need to specify the path to the dataset.
- To run the experiment, please run the following command:

    ```shell
    # require 60GB+ GPU memory
    bash example/run_amazon2M.sh \
    ```

    ```shell
    # require 20GB+ GPU memory
    bash example/run_arxiv.sh.sh \
    ```

    ```shell
    # require 20GB+ GPU memory
    bash example/run_protein.sh \
    ```



- The proposed method is defined in 
  [`hyperbolic_transformer.py`](hyperformer.py)

- The proposed layers is defined in
  [`manifolds/layers.py`](manifolds/layer.py)
