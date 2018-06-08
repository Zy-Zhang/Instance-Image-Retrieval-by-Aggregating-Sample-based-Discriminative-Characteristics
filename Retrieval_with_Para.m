retrieval_with_adc

%% Set all parameters we need to use first
%DataName = 'Oxford5k','Paris6k','Holidays','Instre','UKbench'
%Method = 'KNN','One','LLC'
% KNN: K-nn with exponential
% ONE: K-nn with single
% LLC: K-nn with LLC
% Choose if use exchange pca rule or not
% K = 25 Number of K-Nearest Neighbors

QE = 0;% Choose if use query extension
addpath(genpath('./'));% make the function 'yael_nn' useful for searching
addpath(genpath('../../exemplarsvm-master/'));

%% Load all data we need to use 


DataFolder = ['./data/',DataName,'/'];
load([DataFolder,DataName,'_',Feature,'.mat']);% Load Dataset 

switch DataName
    case 'holidays'        
        Query = DataSet.Query;% We can choose different inder of Query
        Data = DataSet.Data;% this is gallery data
        gnd = DataSet.gnd;
    case 'Paris106k'
        Query = DataSet.Query;
        Data = DataSet.Data;
        Data_search = Data(:,1:105000);
        Num_q = size(Query,2);
        Num_g = size(Data,2);
        gnd = DataSet.gnd;
        Num_search = size(Data_search,2);
    case 'Oxford105k'
        Query = DataSet.Query;
        Data = DataSet.Data;
        Data_search = Data(:,1:5063);
        Num_q = size(Query,2);
        Num_g = size(Data,2);
        Num_search = size(Data_search,2);
        gnd = DataSet.gnd;
        Num_qq = size(DataSet.Query,2);
    case 'UKbench'
        Query = DataSet.Query;
        Data = DataSet.Gallery;
        Num_q = size(Query,2);
        Num_g = size(Data,2);
        gnd = DataSet.gnd;
    otherwise
        if strcmp(Feature,'rMatch_512')
            Query = cell2mat(DataSet.Query);
            Data = cell2mat(DataSet.Data);
            Data_search = [Data(:,1:15000),Data(:,20001:105000)];
            Num_g = length(DataSet.Data);
            Num_q = length(DataSet.Query);
        else
            Query = DataSet.Query;
            Data = DataSet.Data;
            Data_search = DataSet.Data; 
            Num_g = size(Data,2);
            Num_q = size(Query,2);
            Num_qq = size(DataSet.Query,2);
        end
        gnd = DataSet.gnd;% Looking for number of ture positive samples       
        Num_search = size(Data_search,2);
end
Dim = size(Query,1);

%% Strat to Retrieval
%% Prepare Parameters: SVM_q

load([DataFolder,'ExSVM_',DataName,'_',Feature,'.mat']);% Load ExSVM
%load([DataFolder,'Pc_',DataName,'_',Feature,'.mat']);
SVM_q = zeros(Dim,size(Data,2));
for i = 1:size(Data,2)
    SVM_q(:,i) = exsvm{i}.w./norm(exsvm{i}.w);%ex_range(1,i)
end
clear exsvm;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Original Ranking with cos
%This is for AQE
[rank_o,s_o] = yael_nn(Data,-Query,size(Data,2),16);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch Method
    case 'AA'
        Num_qe = 10;
        Neg_c = 30;
        C = 1;
        for iw = 1:size(Query,2)
            data = double([Data(:,rank_o(1:Num_qe,iw)),Data(:,rank_o(end-Neg_c+1:end,iw))]);
            label = double([ones(Num_qe,1);-ones(Neg_c,1)]);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %G = [gnd(iw).ok,gnd(iw).junk];
            %G = [gnd(iw).ok];
            %Index = G(1,1:ceil(Pe*length(G)/10));
            %data = double([Data(:,Index),Data(:,rank_o(end-(Neg_c-1):end,iw))]);
            %label = double([ones(length(Index),1);-ones(Neg_c,1)]);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
            svm_model = libsvmtrain(label,data',sprintf(['-s 0 -t 0 -c %f -w1 %.9f -q'], 0.01, C));
            W = full(sum(svm_model.SVs .* repmat(svm_model.sv_coef,1,size(svm_model.SVs,2)),1));
            DQE(:,iw) = W';
        end
        S_dqe = Data'*DQE;
        [~,rank_dqe] = sort(S_dqe,'descend');
        for num = 1:size(Query,2)
            Map_dqe(1,num) = compute_map(rank_dqe(:,num),gnd(num));
        end
        map_dqe = compute_map(rank_dqe,gnd);%% TP -- DQE
    otherwise
        map_dqe = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[rank_d,~] = yael_nn(Data,-Data,150,16);
%fprintf('The MAP-DQE is: %.2f\n',map_dqe*100);
%for iq = 1:size(Query,2)
    %Query(:,iq) = mean([Query(:,iq),Data(:,rank_o(1:Num_qe,iq))],2);
%end

%% Prepare Parameters: Select

switch Method  % 'TP','EXE'
    case 'TP'
        for Num = 1:size(Query,2)            
            %Ind = [gnd(Num).ok,gnd(Num).junk];
            Ind = [gnd(Num).ok];
            Index = Ind(1,1:ceil(Pe*length(Ind)/10));%Ind;
            %Index = rank_o(1:Num_qe,Num);
            svm(:,Num) = mean(SVM_q(:,Index),2);%% TP -- EXE
            %svm(:,Num) = -mean(PC(:,Index),2);%% TP -- PC
            %svm(:,Num) = -mean(Data(:,Index),2);%% TP -- AQE
        end
    case 'EXE'
        B = 0;
        C = 0;
        %T = 5;
        Re_S = 'Ex';% 'Ex','Hn','Rs','Nn',
        QE = 'ExE';%'ExE','PC','AQE'
        for Num = 1:size(Query,2)
            switch Re_S
                case 'Ex'
                    svm_ini = mean(SVM_q(:,rank_o(1:2,Num)),2);
                    sc_ini = -svm_ini'*Data;
                    [~,cho] = sort(sc_ini,'descend');
                    Threshold = 5;
                    if size(SVM_q,1) == 512
                        Th = -T;% -0.4 for rmac_512/ -0.45 for rmac_2048;
                    else
                        Th = -T;
                    end
                    K_ini = max(length(find(s_o(:,Num) < Th )),Threshold);
                    if K_ini > Threshold
                        Index = Select_Exe(rank_d,rank_o(:,Num)',K_ini);
                    else
                        Index = Select_Exe(rank_d,cho,10);
                    end
                case 'Hn'
                    Index = HN(rank_d,rank_o(:,Num),T);
                case 'Rs'
                    Index = RS(rank_d,rank_o(1,Num),T);
                case 'Nn'
                    Index = rank_o(1:T,Num);
            end
            %C = C + sum(ismember(unique(Index),[gnd(Num).ok,gnd(Num).junk]));
            %B = B + length(unique(Index));
            switch QE
                case 'ExE'
                    svm(:,Num) = mean(SVM_q(:,Index),2);
                case 'PC'
                    svm(:,Num) = -mean(PC(:,Index),2);
                case 'AQE'
                    svm(:,Num) = -mean(Data(:,Index),2);
            end
        end
    case 'ORI'
        svm = -Query;
end

%fprintf('The Select Matrix has been calculated!\n');

%% Calculate the Similarity

A = -svm'*Data;
[~,rank] = sort(A','descend');
Accuracy = 0;
%Accuracy = compute_map(rank,gnd);

%for num = 1:size(Query,2)
    %Map_exe(1,num) = compute_map(rank(:,num),gnd(num));
%end
Plot = 0;
if Plot
    plot(1:size(Query,2),Map_dqe);
    hold on;
    plot(1:size(Query,2),Map_exe);
end

switch DataName
    case 'UKbench'
        Accuracy = 4*Accuracy;
        map_dqe = 4*map_dqe;
end


