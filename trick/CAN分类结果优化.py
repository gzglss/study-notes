'''
原论文的代码实现
'''




# 预测结果，计算修正前准确率
y_pred = model.predict(
    valid_generator.fortest(), steps=len(valid_generator), verbose=True
)
y_true = np.array([d[1] for d in valid_data])
acc_original = np.mean([y_pred.argmax(1) == y_true])
print('original acc: %s' % acc_original)

# 评价每个预测结果的不确定性
k = 3
y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
y_pred_uncertainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)

# 选择阈值，划分高、低置信度两部分
threshold = 0.9
y_pred_confident = y_pred[y_pred_uncertainty < threshold]
y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]
y_true_confident = y_true[y_pred_uncertainty < threshold]
y_true_unconfident = y_true[y_pred_uncertainty >= threshold]

# 显示两部分各自的准确率
# 一般而言，高置信度集准确率会远高于低置信度的
acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean()
acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean()
print('confident acc: %s' % acc_confident)
print('unconfident acc: %s' % acc_unconfident)

# 从训练集统计先验分布
prior = np.zeros(num_classes)
for d in train_data:
    prior[d[1]] += 1.

prior /= prior.sum()

# 逐个修改低置信度样本，并重新评价准确率
right, alpha, iters = 0, 1, 1
for i, y in enumerate(y_pred_unconfident):
    Y = np.concatenate([y_pred_confident, y[None]], axis=0)
    for j in range(iters):
        Y = Y**alpha
        #此处应该是求平均，但是取和的结果也是等价的，因为会在后面对结果进行归一化
        Y /= Y.sum(axis=0, keepdims=True)
        Y *= prior[None]
        Y /= Y.sum(axis=1, keepdims=True)
    y = Y[-1]
    if y.argmax() == y_true_unconfident[i]:
        right += 1

# 输出修正后的准确率
acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)
print('new unconfident acc: %s' % (right / (i + 1.)))
print('final acc: %s' % acc_final)