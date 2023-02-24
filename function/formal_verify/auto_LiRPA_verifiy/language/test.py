
from third_party.auto_LiRPA.auto_LiRPA_verifiy.language.verify import verify
from third_party.auto_LiRPA.auto_LiRPA_verifiy.language.data_utils import get_sst_data
from third_party.auto_LiRPA.auto_LiRPA_verifiy.language.lstm import get_lstm_demo_model


if __name__ == '__main__':
    lstm_model = get_lstm_demo_model()
    ver_data, _ = get_sst_data()
    n_class = 2
    device = 'cpu'
    input_param = {'model': lstm_model,
                   'dataset': ver_data,
                   'n_class': n_class,
                   'up_eps': 0.1,
                   'down_eps': 0.01,
                   'steps': 5,
                   'device': device,
                   'output_path': 'output'}

    result = verify(input_param)
    print(result)

