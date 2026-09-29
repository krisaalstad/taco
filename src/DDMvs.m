function D=DDMvs(t,P,T,a,b,c,D0)
%% DDMs: A very simple degree day seasonal snow model
% Runs potentially large ensemble for a single point/cell

% Hyperparameters:
Tr=3; % Temperature threshold for rainfall (degrees C)
Ts=-1; % Temperature threshold for snowfall (degrees C)
Tm=0; % Temperature threshold for snowmelt (degrees C)
k2c=273.15;

% Pre-allocation
Nt=numel(t); % Number of time steps.
Ne=numel(b);
if isempty(D0)
    Dold=0;
else
    Dold=D0;
end
D=zeros(Nt,Ne);
dTrs=Tr-Ts;
%issnow=Dold>0;

for j=1:Nt
    Tj=T(j)-k2c;
    Tj=Tj-b;
    ddj=Tj-Tm;
    Mj=max(a.*ddj,0);
    frj=(Tj-Ts)./(dTrs);
    frj=max(frj,0);
    frj=min(frj,1);
    fsj=1-frj;
    Sj=(c.*fsj).*(P(j));
    NAj=Sj-Mj;

    % Update
    Dj=max(Dold+NAj,0);
    D(j,:)=Dj;
    Dold=Dj;
end

end