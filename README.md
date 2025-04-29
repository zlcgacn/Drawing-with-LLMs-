# Kaggle Competition: Text-to-SVG Image Generation using Large Language Models (LLM)

This project aims to participate in a Kaggle competition where the core task is to generate Scalable Vector Graphics (SVG) code based on textual descriptions provided by users. The project leverages the text generation capabilities of Large Language Models (LLMs) to create images that match the given descriptions.

## Project Goal

Automatically generate valid SVG code corresponding to an input text description (e.g., "a white cloud floating in a blue sky").

## Methodology and Workflow

This project is implemented within a Kaggle Notebook environment and primarily involves the following steps:

1.  **Environment Setup and Data Loading (Step 1)**
    *   Import necessary Python libraries such as `pandas` (data processing), `numpy` (numerical computation), `os` (file system interaction), and `matplotlib` (data visualization).
    *   Load the training (`train.csv`) and testing (`test.csv`) datasets provided by the competition.
    *   Perform initial analysis and visualization of the description text lengths in the training data to understand data characteristics.

2.  **Large Language Model (LLM) Loading and Configuration (Step 2)**
    *   Import `AutoTokenizer`, `AutoModelForCausalLM`, and `pipeline` from the `transformers` library.
    *   Specify the pre-trained LLM name to be used (e.g., `bigcode/starcoderbase` or `NousResearch/Nous-Hermes-3-Llama-3-8B-preview`, suitable for code generation).
    *   Load the corresponding model's Tokenizer.
    *   Load the pre-trained model and configure it for optimization within the Kaggle environment:
        *   Use `torch.float16` (FP16) to reduce memory footprint and potentially speed up computation.
        *   Enable `low_cpu_mem_usage`.
        *   Set `device_map="auto"` to allow `transformers` to automatically distribute model layers across available hardware (CPU/GPU).
    *   Create a `text-generation` type `pipeline` to simplify subsequent text generation calls.
    *   Define an example `sample_prompt` that clearly instructs the LLM to generate SVG code, specifying requirements (e.g., must include `<svg>` tags, must contain specific elements, no extra explanations).
    *   Demonstrate how to use the `pipeline` for text generation, setting parameters like `max_length`, `num_return_sequences`, `do_sample`, `top_k`, `top_p` to control the diversity and quality of the output.

3.  **Defining the Prediction Class and Prompt Engineering (Step 3)**
    *   Define a `Model` class to encapsulate the prediction logic.
    *   In the `__init__` method, potentially set parameters like a maximum byte size limit for the generated SVG code (`max_svg_bytes`).
    *   Implement the `predict` method:
        *   Accepts a text `prompt` as input.
        *   Constructs a more detailed, structured `full_prompt` containing explicit instructions to guide the LLM in generating compliant SVG code (e.g., emphasizing validity, byte limits, code-only output).
        *   Calls the previously created `llm_generator` (pipeline) for text generation, potentially adjusting parameters like `max_new_tokens`, `temperature`.
        *   Uses regular expressions (`re.search`) to extract the code between `<svg>` and `</svg>` tags from the LLM's generated output.
        *   **Post-processing and Validation**:
            *   If the regex match fails, potentially use the raw generated text or return an error/empty value.
            *   Check if the byte size of the extracted SVG code exceeds the preset limit. If it does, print a warning (the code snippet doesn't explicitly show handling for oversized code, which might require further refinement like truncation or regeneration attempts).
    *   Demonstrate how to instantiate the `Model` class and call the `predict` method to generate SVG.

4.  **Inference on Test Set and Submission File Generation (Step 4)**
    *   Import the `tqdm` library to display progress bars.
    *   Load the test dataset (`test.csv`).
    *   Initialize a list `predictions` to store the results.
    *   Iterate through each row of the test dataset:
        *   Get the `id` and `description`.
        *   Call the `my_model.predict(description)` method to generate the SVG code.
        *   Append a dictionary containing the `id` and the generated `svg_code` to the `predictions` list.
    *   Use `pandas` to convert the `predictions` list into a DataFrame conforming to the Kaggle submission format (with `id` and `prediction` columns).
    *   Save the DataFrame to a `submission.csv` file, setting `index=False`.

## Key Libraries Used

*   `transformers`: For loading and using pre-trained LLM models and tokenizers.
*   `torch`: PyTorch library, one of the backends for `transformers`, used for model computation and GPU acceleration.
*   `pandas`: For data loading, manipulation, and creating the submission file.
*   `numpy`: For numerical operations.
*   `matplotlib`: For data visualization (e.g., analyzing description length distribution).
*   `re`: For regular expression matching to extract SVG code.
*   `tqdm`: For displaying progress bars.

## How to Run

1.  Open this project in a Kaggle Notebook environment.
2.  Ensure the competition data is correctly mounted under the `/kaggle/input/` directory, containing `train.csv` and `test.csv`.
3.  Execute all code cells in the notebook sequentially.
4.  The generated `submission.csv` file will be saved in the `/kaggle/working/` directory and can be directly used for submission.

## Potential Improvements

*   **Stronger SVG Validation**: Use dedicated SVG parsing libraries (like `xml.etree.ElementTree` or third-party libraries) to verify if the generated code is well-formed and valid SVG, beyond just checking byte size.
*   **Error Handling**: Implement more robust handling for cases where the LLM fails to generate, regex extraction fails, or validation fails.
*   **Prompt Optimization**: Further iterate and refine the `full_prompt` to improve the accuracy, consistency, and compliance of the generated SVG.
*   **Model Selection/Fine-tuning**: Experiment with different LLMs or fine-tune a model on a relevant dataset (if permissible and data is available).
*   **Parameter Tuning**: Adjust generation parameters like `temperature`, `top_k`, `top_p`, `max_new_tokens` to find the optimal balance.
*   **Post-processing Logic**: Implement smarter handling logic if SVG code is oversized or invalid, such as attempting truncation, repair, or requesting regeneration.
