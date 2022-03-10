%% PA 8 Diode Parameter Extraction


clear all;
close all;
set(0,'DefaultFigureWindowStyle','docked');

%% 
% set up initial variables
Is = 0.01e-12;      % Forward bias saturation current
Ib = 0.1e-12;       % Breakdown saturation current
Vb = 1.3;           % Breakdown Voltage
Gp = 0.1;           % Parasitic parallel conductance

V = linspace(-1.95,0.7,200);
I = Is.*(exp((1.2/0.025).*V)-1) + Gp.*V - Ib.*(exp((-1.2/0.025).*(V+Vb))-1);


%% Part 1
noise = (rand(1,200)-0.5)./5;
noise = I .* noise;
noise_I = I + noise;

figure(1)
subplot(2,1,1)
hold on
plot(V,noise_I)
title('noise using plot function');
subplot(2,1,2)
hold on
semilogy(V,abs(noise_I))        % abs used because negative data is ignored
title('noise using semilogy function');

%% Part 2
poly_4 = polyfit(V, noise_I, 4);
poly_8 = polyfit(V, noise_I, 8);
I_4 = polyval(poly_4, V);
I_8 = polyval(poly_8, V);

subplot(2,1,1)
plot(V, I_4)
plot(V, I_8)

subplot(2,1,2)
semilogy(V, abs(I_4));
semilogy(V, abs(I_8));

%% Part 3
foa = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ffa = fit(transpose(V),transpose(I),foa);
Ifa = ffa(V);

subplot(2,1,1)
plot(V,Ifa)
subplot(2,1,2)
semilogy(V, abs(Ifa));

fob = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ffb = fit(transpose(V),transpose(I),fob);
Ifb = ffb(V);

subplot(2,1,1)
plot(V,Ifb)
subplot(2,1,2)
semilogy(V, abs(Ifb));

foc = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ffc = fit(transpose(V),transpose(I),foc);
Ifc = ffc(V);

subplot(2,1,1)
plot(V,Ifc)
subplot(2,1,2)
semilogy(V, abs(Ifc));

% In observing the different plots that come from different levels of fit
% paramters, it appears that setting the A,B,C,D values to their values
% from equation 1 improves precision, while setting them as variables makes
% the plot less precise.

%% Part 4
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

subplot(2,1,1)
plot(V,Inn)
legend('plot function', 'poly fit 4th order', 'poly fit 8th order', 'fit part a', 'fit part b', 'fit part c', 'Neural Net Model');
subplot(2,1,2)
semilogy(V, abs(Inn));
legend('plot function', 'poly fit 4th order', 'poly fit 8th order', 'fit part a', 'fit part b', 'fit part c', 'Neural Net Model');

% Upon examination of the plots from the neural net model, it appears that
% the deep learning network creates a plot that is very accurate. Compared
% to the plot function, the neural net model is virtually identical.
