%% paternoster
clc 
clear
close all

%% 
sigmoid = @(x)(1./(1+exp(-x)))


x = -1:.1:2;
y = max(min(x, 1), 0);
plot(x, [sigmoid(4*x-2); y])
grid on
% ylim([0, 1])