%%Plotting value function for chakra
clear;clc;close all;
s1 = linspace (-1,1, 1000);
s2 = linspace (-1,1, 1000);
%% Converge value of w
w= [-4.90037565, -4.85346523, -5.01798707];
[S1,S2]=meshgrid(s1,s2);
%% calculating value function
for i=1:1000
    for j=1:1000   
    V(i,j)=w*([S1(i,j)^2,S2(i,j)^2,1])';
    end
end
%% Plotting value function for chakra
 contour3(S1,S2,V,50) 
xlabel('s_1')
ylabel('s_2')
zlabel('v(s_1,s_2,w) ')
title('Value function for chakra')
