%% retrieval with/without different kinds of query expansino methods on datasets
% the sample provide code running on three datasets which are oxford5k
% paris6k and instre.
% Three kinds of query expansion methods are offered: Average Query
% Expansion, Discriminative Query Expansion, Aggregating Sample-based
% Discriminative Characteristics.

clear all
clc

dataset = 'instre'; % 'oxford5k','paris6k','instre'
query_expansion_method = 'ASDC';% 'AQE','DQE','ASDC'
features = 'resnet';% 'siamac','resnet'

% load neccesary data
load(['./data/',dataset,'_',features,'.mat']);

% calculate the similarity and obtain ranking result with original
% retrieval
s = DataSet.Data'*DataSet.Query;
[~,rank] = sort(s,'descend');
map = compute_map(rank,DataSet.gnd);
fprintf('The original retrieval result is: %.2f(mAP).\n',map*100);

% do the query expansion process

switch query_expansion_method
    case 'AQE'
        Top_rank = 10;
        for q = 1:size(DataSet.Query,2)
            Query(:,q) = mean([DataSet.Data(:,rank(1:Top_rank,q)),DataSet.Query(:,q)],2);
            Query(:,q) = Query(:,q)./norm(Query(:,q));
        end
        s_aqe = DataSet.Data'*Query;
        [~,rank_aqe] = sort(s_aqe,'descend');
        map_aqe = compute_map(rank_aqe,DataSet.gnd);
        fprintf('The AQE result is: %.2f(mAP).\n',map_aqe*100);
    case 'DQE'
        Top_rank = 10;
        Bottom_rank = 30;
        esvm_compile;
        for q = 1:size(DataSet.Query,2)
            data = double([DataSet.Data(:,rank(1:Top_rank,q)),DataSet.Data(:,rank(end-Bottom_rank+1:end,q))]);
            label = double([ones(Top_rank,1);-ones(Bottom_rank,1)]);
            svm_model = libsvmtrain(label,data',sprintf(['-s 0 -t 0 -c %f -w1 %.9f -q'], 0.01, 1));
            W = full(sum(svm_model.SVs .* repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
            Query(:,q) = W';
        end
        s_dqe = DataSet.Data'*Query;
        [~,rank_dqe] = sort(s_dqe,'descend');
        map_dqe = compute_map(rank_dqe,DataSet.gnd);
        fprintf('The DQE result is: %.2f(mAP).\n',map_dqe*100);
    case 'ASDC'
        load(['./data/exsvm_',dataset,'_',features,'.mat']);
        SVM_q = zeros(size(DataSet.Data));
        for i = 1:size(DataSet.Data,2)
            SVM_q(:,i) = exsvm{i}.w./norm(exsvm{i}.w);%ex_range(1,i)
        end
        clear exsvm;
        sd = DataSet.Data'*DataSet.Data;
        [~,rank_d] = sort(sd,'descend');
        for q = 1:size(DataSet.Query,2)
            svm_ini = mean(SVM_q(:,rank(1:2,q)),2);
            sc_ini = -svm_ini'*DataSet.Data;
            [~,cho] = sort(sc_ini,'descend');
            Threshold = 5;
            switch features
                case 'siamac'
                    Radius = 0.4;
                otherwise
                    Radius = 0.45;
            end
            K_ini = max(length(find(s(:,q) > Radius )),Threshold);
            if K_ini > Threshold
                Index = Modified_HN(rank_d(1:150,:),rank(:,q)',K_ini);
            else
                Index = Modified_HN(rank_d(1:150,:),cho,10);
            end
            Query(:,q) = -mean(SVM_q(:,Index),2);
        end
        s_asdc = DataSet.Data'*Query;
        [~,rank_asdc] = sort(s_asdc,'descend');
        map_asdc = compute_map(rank_asdc,DataSet.gnd);
        fprintf('The ASDC result is: %.2f(mAP).\n',map_asdc*100);
end