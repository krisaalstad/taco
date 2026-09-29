%% Run a particle filter and particle batch smoother experiment
% Recreating Figure 4 in the paper. 

clearvars;
rng(12);
p.Ne=1e4;
close all; 


% Add src to the path if needed.
if ~any(which('PBS'))
    paths;
end

% input data originally from https://zenodo.org/records/22938686
% courtesy of NVE for the Filefjell site.
load('input/Snow/FF/FF_forcing.mat');
load('input/Snow/FF/FF_obs.mat');

wyear=2022; 
tstart=datenum(sprintf('01-Sep-%d',wyear-1));
tend=datenum(sprintf('01-Sep-%d',wyear))-1;
thesef=f.t>=tstart&f.t<=tend;
T=f.T(thesef);
P=f.P(thesef);
t=f.t(thesef);

theseo=obs.t>=tstart&obs.t<=tend;
Dt=obs.D(theseo);
tt=obs.t(theseo);


mos=1:6; 
to=zeros(numel(mos),1);
for j=1:numel(mos)
    if mos(j)<9
        to(j)=datenum(sprintf('1.0%d.%d',mos(j),wyear),'dd.mm.yyyy');
    else
        to(j)=datenum(sprintf('1.0%d.%d',mos(j),wyear-1),'dd.mm.yyyy');
    end
end


tam=[9:12 1:9];
tax=zeros(numel(tam),1);
for j=1:numel(tam)
    if j<5
        tax(j)=datenum(sprintf('01.0%d.%d',tam(j),wyear-1),'dd.mm.yyyy');
    else
        tax(j)=datenum(sprintf('01.0%d.%d',tam(j),wyear),'dd.mm.yyyy');
    end
end



Do=zeros(numel(to),1);
sigy=10; 
for j=1:numel(to)
    adtj=abs(tt-to(j));
    ttj=tt(adtj==min(adtj));
    ttj=min(ttj);
    herej=tt==ttj;
    Do(j)=max(Dt(herej)+sigy*randn(1,1),0);
end

hereo=zeros(numel(t),1,'logical');
for j=1:numel(t)
    if any(t(j)==to)
        hereo(j)=1;
    end
end


meda=4;
siga=0.2;
a=exp(log(meda)+siga.*randn(1,p.Ne));

mub=0; sigb=1;
b=mub+sigb.*randn(1,p.Ne);

medc=1; sigc=0.5;
c=exp(log(medc)+sigc.*randn(1,p.Ne));

%% Filtering experiment
af=a;
bf=b;
cf=c;
Dfpri=[];
Dfpost=[];
tw=[to; t(end)]; % End of each window
for j=1:numel(tw)
    if j==1
        D0=0;
        twold=t(1)-1;
    else
        D0=Dj(end,:);
        twold=tw(j-1);
    end
    herej=t>twold&t<=tw(j);
    Tj=T(herej);
    Pj=P(herej);
    tj=t(herej);

    Dj=DDMvs(tj,Pj,Tj,af,bf,cf,D0);
    Dfpri=[Dfpri; Dj];
    
    if j<=numel(to)
        w=PBS( Dj(end,:),Do(j),sigy^2);
    end

    inds=(1:p.Ne)';
    reinds=randsample(inds,p.Ne,true,w);

    % For illustration, running without jitter just resampling.
    % Causes degeneracy.
    dopert=0; % Jitter switch.
    if dopert
        af=exp(log(af(reinds))+0.1.*randn(1,p.Ne));
        bf=bf(reinds)+0.1.*randn(1,p.Ne);
        cf=exp(log(cf(reinds))+0.1.*randn(1,p.Ne));
    else
        af=af(reinds);
        bf=bf(reinds);
        cf=cf(reinds);
    end
    Dj=Dj(:,reinds); % Update Dj for next window (also prior)
    Dfpost=[Dfpost; Dj];
end


%% Smoothing experiment
as=a;
bs=b;
cs=c;
Dspri=[];
Dspost=[];
tw=t(end); % End of each window
for j=1:numel(tw)
    if j==1
        D0=0;
        twold=t(1)-1;
    else
        D0=Dj(end,:);
        twold=tw(j-1);
    end
    herej=t>twold&t<=tw(j);
    Tj=T(herej);
    Pj=P(herej);
    tj=t(herej);

    Dj=DDMvs(tj,Pj,Tj,as,bs,cs,D0);
    Dspri=[Dspri; Dj];
    
    if j<=numel(to)
        w=PBS( Dj(hereo,:),Do(:),sigy^2);
    end

    inds=(1:p.Ne)';
    reinds=randsample(inds,p.Ne,true,w);

    as=exp(log(as(reinds)));
    bs=bs(reinds);
    cs=exp(log(cs(reinds)));
    Dj=Dj(:,reinds); % Update Dj for next window (also prior)
    Dspost=[Dspost; Dj];
end
%%

pcts=[2.5 97.5];
fs=30;
fis=figure('units','inch','position',[0,0,18,18]);
pause(2);

tld = tiledlayout(2,1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
lwt=0.5;
for j=1:numel(to)
    ptt=plot([to(j) to(j)],[0 500],'Color',[0.5 0.5 0.5 0.5],...
        'LineWidth',1,'LineStyle','-'); hold on;
    if j<6
        pt(j)=ptt;
    end
end


percplot(t,prctile(Dfpri,pcts,2),[0.8 0 0],0.2,'-',[0.8 0 0]); hold on;
percplot(t,prctile(Dfpost,pcts,2),[0 0 0.8],0.2,'-',[0 0 0.8]);
pt(1)=plot(t,mean(Dfpri,2),'Color',[0.8 0 0 0.6],'LineWidth',2);
pt(2)=plot(t,mean(Dfpost,2),'Color',[0 0 0.8 0.6],'LineWidth',2);

postm=mean(Dfpost,2);
rmse=sqrt(mean((postm-Dt).^2));
fprintf('\n Filtering rmse %4.2f',rmse);

pt(3)=plot(tt,Dt,'Color',[0 0 0],'LineWidth',4.*lwt,'LineStyle',':');
pt(4)=scatter(to,Do,150,'o','MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0.95 0.8 0.2],'LineWidth',2);
ylim([0 500]);
set(gca,'TickDir','out','LineWidth',2,'TickLength',[0.0025, 0.005]);
set(groot, 'defaultAxesTickLabelInterpreter','LaTex');
set(gca,'TickLabelInterpreter','LaTex');
set(gca,'GridLineStyle',':','FontSize',0.8.*fs);
datetick('x','keeplimits','keepticks');
xlim([datenum(sprintf('01-Oct-%d',wyear-1)) ...
    datenum(sprintf('01-Jul-%d',wyear))]);
set(gca,'XTick',tax,'XTickLabels',[]);
title('Particle Filter','Interpreter','Latex','FontSize',fs);
ylabel('SWE, $m$ [kg m$^{-2}$]','Interpreter','Latex','FontSize',fs)


leg=legend(pt,{'Prior';...
    'Posterior';...
    'Ground truth';...
    'Observations';...
    'Windows'},'Interpreter','Latex','Location','NorthWest','FontSize',fs);
nexttile;
for j=numel(to):numel(to)
    plot([to(j) to(j)],[0 500],'Color',[0.5 0.5 0.5 0.5],...
        'LineWidth',1,'LineStyle','-'); hold on;
end

percplot(t,prctile(Dspri,pcts,2),[0.8 0 0],0.1,'-',[0.8 0 0]); hold on;
percplot(t,prctile(Dspost,pcts,2),[0 0 0.8],0.1,'-',[0 0 0.8]);
pt(1)=plot(t,mean(Dspri,2),'Color',[0.8 0 0 0.6],'LineWidth',2);
pt(2)=plot(t,mean(Dspost,2),'Color',[0 0 0.8 0.6],'LineWidth',2);
postm=mean(Dspost,2);
rmse=sqrt(mean((postm-Dt).^2));
fprintf('\n Smoothing rmse %4.2f',rmse);
pt(3)=plot(tt,Dt,'Color',[0 0 0],'LineWidth',4.*lwt,'LineStyle',':');
pt(4)=scatter(to,Do,150,'o','MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0.95 0.8 0.2],'LineWidth',2);
ylim([0 500]);
set(gca,'TickDir','out','LineWidth',2,'TickLength',[0.0025, 0.005]);
set(groot, 'defaultAxesTickLabelInterpreter','LaTex');
set(gca,'TickLabelInterpreter','LaTex');
set(gca,'GridLineStyle',':','FontSize',0.8.*fs);
set(gca,'XTick',tax,'XTickLabels',[]);
xlim([datenum(sprintf('01-Oct-%d',wyear-1)) ...
    datenum(sprintf('01-Jul-%d',wyear))]);
datetick('x','keeplimits','keepticks');
title('Particle Batch Smoother','Interpreter','Latex','FontSize',fs)
ylabel('SWE, $m$ [kg m$^{-2}$]','Interpreter','Latex','FontSize',fs)

linkaxes(findall(gcf, 'type', 'axes'), 'x');
ax=gca;
ax.Toolbar.Visible = 'off';

pause(2);
figname='PF_PBS';
toprint=sprintf('%s.pdf',figname);
exportgraphics(tld, toprint, 'ContentType', 'vector');
fis.Visible='off';