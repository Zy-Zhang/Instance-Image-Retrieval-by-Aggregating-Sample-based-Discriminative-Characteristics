function Index1 = Modified_HN(rank_d,rank_q,K_ini)

K_ini = min(K_ini,size(rank_d,1)-1);
R = floor(K_ini/2);
Ind = 1:K_ini;
Index1 = rank_q(1,Ind);
Index2 = ones(1,length(Ind));
Sup = 1;
Num = 0;

while ~isempty(Sup)
    Sup = [];
    for j = find(Index2 == 1)%1:length(Index1)%
        mem = sum(ismember(rank_d(1:K_ini+1,rank_d(1:K_ini+1,Index1(1,j))),Index1(1,j)),1);
        ck_ = find(mem == 1);
        A = ismember(rank_d(ck_,Index1(1,j)),Index1);
        if sum(A) > R+1 || 2*sum(A) >= length(ck_)+1
            sup = rank_d(find(A == 0),Index1(1,j))';
            Sup = [Sup,sup(find(ismember(sup,Sup) == 0))];
            Index2(1,j) = 0;
        else
            Index2(1,j) = 0;
        end
    end
    Index1 = [Index1,Sup];
    Index2 = [Index2,ones(1,length(Sup))];
    K_ini = length(find(Index2 == 1))-3;%-3;
    R = floor(K_ini/2);
    Num = Num + 1;
    if Num == 3 || K_ini >= size(rank_d,1) 
        break;
    end
end