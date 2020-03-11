import random
import torch
import torch.optim as optim
from utils import *
from model import DMMN_SDCM
from evals import *
import argparse


def net_copy(net, copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--embedding_fname', default='/data/linpq/Word2Vec/glove.840B.300d.txt', type=str,
                        help='file name of embeddings')
    parser.add_argument('--dataset', default='data/laptop/', type=str, help='data set')
    parser.add_argument('--pre_processed', default=1, type=int, help='denote whether the data has been pre-processed')

    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    parser.add_argument('--embedding_dim', default=300, type=int, help='dimension of embedding vectors')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--hidden_size', default=50, type=int, help='dimension of output size')
    parser.add_argument('--n_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--early_stopping_num', default=40, type=int, help='number of epochs for early stopping')
    parser.add_argument('--n_class', default=3, type=int, help='number of classes')
    parser.add_argument('--n_hop', default=2, type=int, help='number of layers in the memory network')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--l2_reg', default=0.0001, type=float, help='weight of L2 regularization term')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--cml_weight', default=1.5, type=float, help='weight of context moment learning loss')

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.set_device(args.gpu_id)

    train_fname = args.dataset + '/train.txt'
    test_fname = args.dataset + '/test.txt'
    data_info = args.dataset + '/data_info.txt'
    train_data = args.dataset + '/train_data.txt'
    test_data = args.dataset + '/test_data.txt'
    analysis_fname = args.dataset + '/analysis.csv'

    print('Loading data info ...')
    word2id, max_sentence_len, max_aspect_len, max_aspect_num = get_data_info(train_fname, test_fname, data_info,
                                                                              args.pre_processed)

    print('Loading training data and testing data ...')
    train_data = read_data(train_fname, word2id, max_sentence_len, max_aspect_len, max_aspect_num, train_data,
                           args.pre_processed)
    test_data = read_data(test_fname, word2id, max_sentence_len, max_aspect_len, max_aspect_num, test_data,
                          args.pre_processed)

    print('Loading pre-trained word vectors ...')
    embedding_matrix = load_word_embeddings(args.embedding_fname, args.embedding_dim, word2id)

    print('Building torch model ...')
    model = DMMN_SDCM(len(word2id), args.embedding_dim, embedding_matrix, args.hidden_size, args.n_hop).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
    best_acc, best_f, best_sl, best_epoch, es_cnt = 0.0, 0.0, 0.0, 0, 0
    best_model = DMMN_SDCM(len(word2id), args.embedding_dim, embedding_matrix, args.hidden_size, args.n_hop).cuda()
    best_y_pred = []

    print('Training ...')
    for i in range(args.n_epoch):
        cost, cnt, total_correct_num = 0.0, 0, 0
        train_y_pred, train_y_gold = [], []
        for sample, _ in get_batch_data(train_data, args.batch_size, True):
            loss, sloss, num, correct_num, y_pred, y_gold = model.forward(sample, args.dropout)
            optimizer.zero_grad()
            tloss = loss + args.cml_weight * sloss
            tloss.backward()
            optimizer.step()
            cost += tloss.item() * num
            cnt += num
            total_correct_num += correct_num.item()
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
        train_loss = cost / cnt
        train_acc, train_f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)

        cost, cnt, total_correct_num = 0.0, 0, 0
        test_y_pred, test_y_gold = [], []
        for sample, _ in get_batch_data(test_data, args.batch_size, False):
            _, _, num, correct_num, y_pred, y_gold = model.forward(sample, 0)
            total_correct_num += correct_num.item()
            test_y_pred.extend(y_pred)
            test_y_gold.extend(y_gold)
        test_acc, test_f, _, _ = evaluate(pred=test_y_pred, gold=test_y_gold)

        print(
            'Epoch %d, Train Loss %.5f, Train Acc %.5f, Train F1 %.5f, Test Acc %.5f, Test F1 %.5f' % (
                i, train_loss, train_acc, train_f, test_acc, test_f))
        if test_acc + test_f > best_acc + best_f:
            best_acc = test_acc
            best_f = test_f
            best_epoch = i
            best_y_pred = [pred for pred in test_y_pred]
            net_copy(best_model, model)
            es_cnt = 0
        es_cnt += 1
        if es_cnt >= args.early_stopping_num:
            break
    print('The Best Result: Acc %.5f, F1 %.5f, Epoch %d' % (best_acc, best_f, best_epoch))
    torch.save(best_model, './models/laptop_best_model')
    save_analysis_result(test_fname, np.array(best_y_pred), analysis_fname)
