from sklearn import metrics

def evaluate(target, predict):
	print(f"Adjusted mutual info score: {metrics.adjusted_mutual_info_score(target, predict)}")
	print(f"Adjusted rand score: {metrics.adjusted_rand_score(target, predict)}")
	#print(f"{metrics.calinski_harabasz_score(predict, target)}")
	print(f'Completeness score: {metrics.completeness_score(target, predict)}')
	#print(f'{metrics.cluster.contingency_matrix(target, predict)}')
	#print(f"{metrics.cluster.pair_confusion_matrix(target, predict)}")
	print(f"Fowlkes mallow score: {metrics.fowlkes_mallows_score(target, predict)}")
	a, b, c = metrics.homogeneity_completeness_v_measure(target, predict)
	print(f"Homogeneity: {a} \nCompleteness score: {b}\nV measure: {c}")
	print(f"Mutual information between two clusterings score: {metrics.mutual_info_score(target, predict)}")
	print(f"Normalized Mutual Information between two clusterings score: {metrics.normalized_mutual_info_score(target, predict)}")
	print(f"Rand score: {metrics.rand_score(target, predict)}")
	#print(f"{}")

