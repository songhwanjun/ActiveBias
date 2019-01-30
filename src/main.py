import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from reader import cifar10_reader
from algorithm.default import *
from algorithm.activebias import *

def main():
    print("------------------------------------------------------------------------")
    print("This code to train densnet(L=40, k=12) on cifar-10 using Default or ActiveBias.")
    print("You can easily train other datasets by modifying ""cifar10_reader.py"".")
    print("\nDescription -----------------------------------------------------------")
    print("For Training, training_epoch = 200, batch = 128, initial_learning rate = 0.1 (decayed 50% and 75% of total number of epochs)")
    print("               use momentum of 0.9, warm_up=15, smoothness=0.2")
    print("You can easily change the value in main.py")

    if len(sys.argv) != 5:
        print("Run Cmd: python main.py  gpu_id  method_name   noise_rate  log_dir")
        print("\nParamters -----------------------------------------------------------")
        print("gpu_id: gpu number which you want to use")
        print("method: {Default, ActiveBias}")
        print("noise_rate: the rate which you want to corrupt")
        print("log_dir: log directory to save training and test errors")
        sys.exit(-1)

    input_reader = cifar10_reader.ImageReader()
    input_reader.maybe_download_and_extract()

    # For user parameters
    gpu_id = int(sys.argv[1])
    method_name = sys.argv[2]
    noise_rate = float(sys.argv[3])
    log_dir = sys.argv[4]

    # For fixed parameters
    optimizer = 'momentum'
    total_epochs = 200
    batch_size = 128
    lr_boundaries = [40000, 60000]
    lr_values = [0.1, 0.02, 0.004]
    warm_up = 15
    smoothness = 0.2

    if method_name == "Default":
        default(gpu_id, input_reader, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, log_dir=log_dir)
    elif method_name == "ActiveBias":
        active_bias(gpu_id, input_reader, total_epochs, batch_size, lr_boundaries, lr_values, optimizer, noise_rate, warm_up, smoothness, log_dir=log_dir)

if __name__ == '__main__':
    print(sys.argv)
    main()
