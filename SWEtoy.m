clearvars;
close all;
rng(12345);


% Add src to the path if needed.
if ~any(which('PBS'))
    paths;
end

H=@(M,r) (M./r); % Observation operator

dtrue=1.2; % True snow depth
rtrue=300; % True density
Mtrue=rtrue*dtrue; % True SWE

sigd=0.1; % Snow depth measurement error standard deviation (m)
dobs=dtrue-1*sigd;

% Prior sampling
Ne=3e1;

Mm=500;
Ms=100;

M=Mm+Ms*randn(Ne,1);

rm=250;
rs=50;

r=rm+rs*randn(Ne,1);

dpred=H(M,r);

xls=[200 800]; % 3 sigma range
yls=[100 400]; % 3 sigma range


%% Grid approximation
dr=2.5;
ra=dr:dr:500;

dM=2.5;
Ma=dM:dM:1e3;


[Mg,rg]=meshgrid(Ma,ra);

Dg=H(Mg,rg);

logpriMg=-0.5.*(((Mg-Mm).^2)./(Ms^2))-0.5*log(2*pi*Ms^2);
logprirg=-0.5.*(((rg-rm).^2)./(rs^2))-0.5*log(2*pi*rs^2);
logprig=logpriMg+logprirg;

loglikeg=-0.5.*(((dobs-Dg).^2)./(sigd^2))-0.5*log(2*pi*sigd^2);

logjointg=logprig+loglikeg;

evig=trapz(Ma',trapz(ra',exp(logjointg),1),2);

logpostg=logjointg-log(evig);


% Control calculation to verify everything integrates to 1
I=trapz(Ma',trapz(ra',exp(logprig),1),2);

%% Figure 3

fis=figure('units','inch','position',[0,0,18,18],'Renderer','opengl');
pause(2);

load('input/batlowW.mat');
cmap=batlowW;
cmap=flipud(cmap);
colormap(cmap);

fs=25;
r2a=1;
thresh=0;
tld = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;

prig=exp(logprig);
imagesc(Ma,r2a.*ra,prig,'AlphaData',prig>thresh); axis xy; axis square;
set(gca,'TickDir','out','LineWidth',2, 'TickLength',[0.0025, 0.0025]);
set(groot, 'defaultAxesTickLabelInterpreter','LaTex');
set(gca,'TickLabelInterpreter','LaTex');
set(gca,'GridLineStyle',':','FontSize',0.8.*fs);
xlabel('Snow water equivalent (SWE), $m$ [kg m$^{-2}$]','Interpreter','Latex',...
    'FontSize',fs);
hold on;

thetapri=[M'; r'];
spri(1)=scatter(thetapri(1,:),r2a.*thetapri(2,:),250,'o','MarkerEdgeColor',...
    [0 0 0],'MarkerFaceColor',[0.8 0 0],'MarkerFaceAlpha',0.6);
hold on;
spri(2)=scatter(Mtrue,rtrue,350,'p','MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[1 1 1]);
xlim(xls);
ylim(yls);
title('(a) Joint prior $p(m,\rho)$',...
    'Interpreter','Latex','FontSize',fs);

nexttile;
postg=exp(logpostg);
imagesc(Ma,r2a.*ra,postg,'AlphaData',postg>thresh); axis xy; axis square;
set(gca,'TickDir','out','LineWidth',2,'TickLength',[0.005, 0.005]);
set(groot, 'defaultAxesTickLabelInterpreter','LaTex');
set(gca,'TickLabelInterpreter','LaTex');
set(gca,'GridLineStyle',':','FontSize',0.8.*fs);
xlabel('Snow water equivalent (SWE), $m$ [kg m$^{-2}$]','Interpreter','Latex',...
    'FontSize',fs);
leg=legend(spri,{'Prior ensemble','Truth'},'Interpreter','Latex','FontSize',fs,...
    'Location','NorthWest');



%% EnKA
thetapri=[M'; r'];

Na=1;
alpha=Na; dostoch=0;

for j=1:Na
    thetapost=EnKA(thetapri,dobs,dpred',alpha,sigd^2,dostoch);
    dpred=H(thetapost(1,:)',thetapost(2,:)');
    thetapri=thetapost;
end

hold on;
spost(3)=scatter(thetapost(1,:),r2a.*thetapost(2,:),250,'d','MarkerFaceColor',[0 0 0.8],...
    'MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.6);

%% PBS
thetapri=[M'; r'];
dpred=H(thetapri(1,:)',thetapri(2,:)');
w=PBS(dpred',dobs,sigd^2);

inds=1:Ne;
reinds=randsample(Ne,Ne,true,w);
spost(1)=scatter(thetapri(1,:),r2a.*thetapri(2,:),250,'o','MarkerEdgeColor',...
    [0 0 0],'MarkerFaceColor',[0.8 0 0],'MarkerFaceAlpha',0.2);
hold on;
spost(4)=scatter(thetapri(1,reinds),r2a.*thetapri(2,reinds),250,'s','MarkerFaceColor',[0 0.8 0.8],...
    'MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.6);
spost(2)=scatter(Mtrue,rtrue,350,'p','MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[1 1 1]);
title('(b) Joint posterior $p(m,\rho \mid d^o)$',...
    'Interpreter','Latex','FontSize',fs);
leg=legend(spost,{'Prior ensemble','Truth','EnKF posterior','Particle posterior'},'Interpreter','Latex','FontSize',fs,...
    'Location','SouthWest');
xlim(xls);
ylim(yls);

linkaxes(findall(gcf, 'type', 'axes'), 'x');
xlim(xls);
ylim(yls);
ax=gca;
ax.Toolbar.Visible = 'off';
ylabel(tld,'Snow density, $\rho$ [m w.e.]','Interpreter','Latex','FontSize',fs);
figname='joint';
toprint=sprintf('%s.jpg',figname);
exportgraphics(tld,toprint,'Resolution',600);

pause(2);
close all;
%% Bayes' for SWE = Figure 2
fis=figure('units','inch','position',[0,0,18,18]);
pause(2);
tld = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
lw=3;
al=0.6;
colororder({'k','k'})

priMg=trapz(ra',prig,1);
pt(4)=plot([Mtrue Mtrue],[0 8e-3],'LineWidth',lw,'Color',[0 0 0],...
    'LineStyle','-.'); hold on;
pt(1)=plot(Ma,priMg,'Color',[0.8 0 0 al],'LineWidth',lw);
apr=area(Ma,priMg);
apr.FaceColor=[0.8 0 0];
apr.FaceAlpha=0.25.*al;
apr.EdgeColor='none';



% integrated likelihood \int p(y,phi|theta) dphi = int p(y|theta,phi)p(phi)
% d\phi = p(y|theta) 
ilike=trapz(ra',exp(loglikeg+logprirg),1);
ev=trapz(Ma',ilike.*priMg,2);

hold on;

set(gca,'TickDir','out','LineWidth',2,'TickLength',[0.0025, 0.005]);
set(groot, 'defaultAxesTickLabelInterpreter','LaTex');
set(gca,'TickLabelInterpreter','LaTex');
set(gca,'GridLineStyle',':','FontSize',0.8.*fs);

postMg=trapz(ra',exp(logpostg),1);

ylabel('Probability density [m$^2$ kg$^{-1}$]','Interpreter','Latex',...
    'FontSize',fs);
hold on;
nc=trapz(Ma',ilike,2);
pt(2)=plot(Ma,ilike./(nc),'Color',[0.3 0.3 0.3 al],'LineWidth',lw);
ali=area(Ma,ilike./nc);
ali.FaceColor=[0.3 0.3 0.3];
ali.FaceAlpha=0.25.*al;
ali.EdgeColor='none';


pt(3)=plot(Ma,postMg,'Color',[0 0 0.8 al],'LineWidth',lw);
apo=area(Ma,postMg);
apo.FaceColor=[0 0 0.8];
apo.FaceAlpha=0.25.*al;
apo.EdgeColor='none';

xlabel('Snow water equivalent (SWE) [kg m$^{-2}$]','Interpreter','Latex',...
    'FontSize',fs);
leg=legend(pt,{'Prior $p(m)$',...
    'Likelihood* $p(d^o\mid m)$',...
    'Posterior $p(m\mid d^o)$ ',...
    'True SWE $m^\star$'},...
    'Location','NorthEast','FontSize',fs,'Interpreter','Latex');
xlim([0,900])
axis square;
title('(a) Hidden state: SWE','Interpreter','Latex','FontSize',fs);

va=ra;
da=0.01:0.01:4; % Same size as M (but not same coordinate due to dependence)
[dg,vg]=meshgrid(da,va);

Mgre=dg.*vg;

logpriMgre=-0.5.*(((Mgre-Mm).^2)./(Ms^2))-0.5*log(2*pi*Ms^2);
logprirvg=-0.5.*(((vg-rm).^2)./(rs^2))-0.5*log(2*pi*rs^2);
logprigre=logpriMgre+logprirg+log(vg); % joint p(d,v)
loglikegre=-0.5.*(((dobs-dg).^2)./(sigd^2))-0.5*log(2*pi*sigd^2);
logjointgre=logprigre+loglikegre;
evigre=trapz(da',trapz(va',exp(logjointgre),1),2);
logpostgre=logjointgre-log(evigre);

nexttile;
lw=3;
al=0.6;
colororder({'k','k'})

prid=trapz(ra',exp(logprigre),1); 
% p(d,v)=p(M,r) |det(J)| = p(M,r) |r| if v=r. Units m^2 kg^-1 

pt(4)=plot([dtrue dtrue],[0 10],'LineWidth',lw,'Color',[0 0 0],...
    'LineStyle','-.'); hold on;
pt(1)=plot(da,prid,'Color',[0.8 0 0 al],'LineWidth',lw);
apr=area(da,prid);
apr.FaceColor=[0.8 0 0];
apr.FaceAlpha=0.25.*al;
apr.EdgeColor='none';

% integrated likelihood \int p(y,phi|theta) dphi = int p(y|theta,phi)p(phi)
% d\phi = p(y|theta) 
ilikere=trapz(va',exp(loglikegre+logprirg),1);

hold on;

set(gca,'TickDir','out','LineWidth',2,'TickLength',[0.0025, 0.005]);
set(groot, 'defaultAxesTickLabelInterpreter','LaTex');
set(gca,'TickLabelInterpreter','LaTex');
set(gca,'GridLineStyle',':','FontSize',0.8.*fs);

postdg=trapz(va',exp(logpostgre),1);

pt(5)=plot([dobs dobs],[0 10],'LineWidth',lw,'Color',[0.3 0.3 0.3 al],...
   'LineStyle',':');
ylabel('Probability density [m$^{-1}$]','Interpreter','Latex',...
    'FontSize',fs);
hold on;
nc=trapz(da',ilikere,2);
pt(2)=plot(da,ilikere./nc,'Color',[0.3 0.3 0.3 al],'LineWidth',lw);
ali=area(da,ilikere./nc);
ali.FaceColor=[0.3 0.3 0.3];
ali.FaceAlpha=0.25.*al;
ali.EdgeColor='none';


pt(3)=plot(da,postdg,'Color',[0 0 0.8 al],'LineWidth',lw);
apo=area(da,postdg);
apo.FaceColor=[0 0 0.8];
apo.FaceAlpha=0.25.*al;
apo.EdgeColor='none';


xlabel('Snow depth [m]','Interpreter','Latex',...
    'FontSize',fs);

leg=legend(pt,{'Prior $p\left(\hat{d}\right)$',...
    'Likelihood* $p\left(d^o\mid \hat{d}\right)$',...
    'Posterior $p\left(\hat{d}\mid d^o\right)$',...
    'True depth $d^\star$',...
    'Observed depth $d^o$'},...
    'Location','NorthEast','FontSize',fs,'Interpreter','Latex');
xlim([0,4])
ylim([0,5])
axis square;
title('(b) Observable: Snow depth','Interpreter','Latex','FontSize',fs);


pause(2);
figname='Bayes';
toprint=sprintf('%s.pdf',figname);
exportgraphics(tld,toprint, 'ContentType', 'vector');

