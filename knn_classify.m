function predictions = knn_classify(k, train_data, train_label, eval_data)
% k-nearest neighbor classifier
% Input:
%  k: number of nearest neighbors
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  eval_data: M*D matrix, each row as a sample and each column as a
%  feature
%
% Output:
%  predictions: predicted labels for evaluation dataset
%
    predictions = zeros(rows(eval_data), 1);
    for i = 1:rows(eval_data)
        dist = sqrt(sumsq((eval_data(i,:) - train_data), 2)); % compute euclid dist

        all = [dist, train_label]; % concat distances to labels
        [sorted, orig] = sortrows(all, 1); % sort by distances
        k_nearest = sorted(1:k, :); % take k min values
        [m, f, c] = mode(k_nearest(:, 2)); % find the mode of the 2nd column

        ties = size(c{1}, 1);
        if (ties > 1)
            class = c{1}(randi(ties)); % randomly select one of the classes
        else
            class = m;
        endif

        predictions(i,:) = class;
    endfor

    return
endfunction
