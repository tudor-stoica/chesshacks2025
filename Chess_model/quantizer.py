import torch
import torch.nn as nn
import chess_cnn  # Assuming this is the .py file with your 'chessCNN' class
import pickle   # <--- 1. Import the pickle module

move_map_PATH = "chess_cnn_move_map.pkl"
MODEL_PATH = 'chess_cnn.pth' 

try:
    # --- 2. Un-pickle the move_map file ---
    with open(move_map_PATH, 'rb') as f:
        move_map = pickle.load(f)
    print(f"Successfully loaded move map with {len(move_map)} moves.")

    # --- 3. Load the fp32 model ---
    # Create an instance of the model structure
    # Note: Using chess_cnn.chessCNN since you imported chess_cnn
    model_fp32 = chess_cnn.chessCNN(len(move_map)+ 1)

    # Load the saved state_dict (weights)
    model_fp32.load_state_dict(torch.load(MODEL_PATH))
    
    # --- 4. Set model to evaluation mode ---
    model_fp32.eval()

    print(f"Successfully loaded model from {MODEL_PATH}")
    print(f"Original model parameter dtype: {next(model_fp32.parameters()).dtype}")

    # --- 5. Convert the model to fp16 ---
    model_fp16 = model_fp32.half()
    print(f"Converted model parameter dtype: {next(model_fp16.parameters()).dtype}")

    # --- 6. (Optional) Move model to GPU ---
    if torch.cuda.is_available():
        model_fp16 = model_fp16.to('cuda')
        print("Model moved to GPU.")
    else:
        print("CUDA (GPU) not available. Running on CPU.")

    # --- 7. Run Inference ---
    # !!! IMPORTANT: Update this shape to match your model's input !!!
    # A chess CNN likely expects (batch, channels, height, width), e.g., (4, 12, 8, 8)
    # The (4, 128) shape from your example is probably incorrect.
    
    # Placeholder for a batch of 4, 12 channels (piece types/history), 8x8 board
    input_data_fp32 = torch.randn(4, 12, 8, 8) 
    print(f"\nUsing DUMMY input data with shape: {input_data_fp32.shape}")

    # Convert the input tensor to fp16
    input_data_fp16 = input_data_fp32.half()

    if torch.cuda.is_available():
        input_data_fp16 = input_data_fp16.to('cuda')

    print(f"Input tensor dtype: {input_data_fp16.dtype}")

    # Perform inference
    with torch.no_grad(): # Disable gradient calculation
        output = model_fp16(input_data_fp16)
        print("--- Inference Successful ---")
        print(f"Output dtype: {output.dtype}")
        print(f"Output shape: {output.shape}") # Should be (4, len(move_map))

except FileNotFoundError as e:
    print(f"Error: File not found.")
    print(f"Details: {e}")
    print(f"Please check your paths: \n- {MODEL_PATH}\n- {move_map_PATH}")
except AttributeError as e:
    print(f"Error: Could not find class 'chessCNN'.")
    print(f"Details: {e}")
    print("Make sure your 'chess_cnn.py' file contains the 'chessCNN' class definition.")
except Exception as e:
    print(f"An error occurred during loading or inference: {e}")