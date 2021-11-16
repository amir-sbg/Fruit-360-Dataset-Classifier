from ANN_Project_Assets import Loading_Datasets
import numpy as np

'''
# Author : Amir Sabbagh Ziarani
# Version: 1.0
# Goal   : Detecting accuracy and validity of model before training
'''

def sigmoid(inpt):
    return 1.0 / (1 + np.exp(-1 * inpt))


def not_trained_200sample():
    train_set_features, train_set_labels, test_set_features, test_set_labels = Loading_Datasets.loadData()

    shuffler = np.random.permutation(len(train_set_features))
    train_set_features = train_set_features[shuffler]
    train_set_labels = train_set_labels[shuffler]

    limit=200

    train_set_features=train_set_features[:limit]
    train_set_labels=train_set_labels[:limit]


    HL1_neurons = 150
    input_plus_bias_size=train_set_features.shape[1]+1
    input_HL1_weights = np.random.uniform(low=0, high=1, size=(input_plus_bias_size, HL1_neurons))


    HL2_neurons = 60
    HL1_neurons_plus_bias_size =HL1_neurons+1
    HL1_HL2_weights = np.random.uniform(low=0, high=1, size=(HL1_neurons_plus_bias_size, HL2_neurons))

    output_neurons = 4
    HL2_neurons_plus_bias_size=HL2_neurons+1
    HL2_output_weights = np.random.uniform(low=0, high=1, size=(HL2_neurons_plus_bias_size, output_neurons))


    bias= np.array([np.zeros(limit)])


    input_plus_bias=np.concatenate((bias.T, train_set_features), axis=1)


    print("input :",np.shape(input_plus_bias) ,"*", np.shape(input_HL1_weights),"->",end="")
    H1_outputs = np.matmul(input_plus_bias, input_HL1_weights)

    H1_outputs = sigmoid(H1_outputs)
    print( np.shape(H1_outputs) )



    HL1_neurons_plus_bias = np.concatenate((bias.T, H1_outputs), axis=1)

    print("HL1   :",np.shape(HL1_neurons_plus_bias),"*", np.shape(HL1_HL2_weights)," ->",end="")
    H2_outputs = np.matmul(HL1_neurons_plus_bias, HL1_HL2_weights)
    H2_outputs = sigmoid(H2_outputs)
    print(np.shape(H2_outputs))


    HL2_neurons_plus_bias = np.concatenate((bias.T, H2_outputs), axis=1)

    print("HL2   :",np.shape(HL2_neurons_plus_bias)," *", np.shape(HL2_output_weights),"   ->",end="")
    out_otuputs = np.matmul(HL2_neurons_plus_bias, HL2_output_weights)
    print(np.shape(out_otuputs))

    predicted_label = np.where(out_otuputs == np.max(out_otuputs))


    accuracy_result=(len(train_set_labels)-np.count_nonzero(predicted_label[1]-train_set_labels))/len(train_set_labels)
    print(accuracy_result)

if __name__ == "__main__":
    not_trained_200sample()
