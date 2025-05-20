function [skeleton]=sortskeleton( skeleton)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[nNode]=size(skeleton, 1);
dGraph=skeleton;
%apporixmate topical sort
idx=zeros(nNode,1);
nodeFlag=true(nNode, 1);
for iPass=1:nNode
    % find the mininmal in-degree variable
    nodeIdx=find(nodeFlag==true);
    for i = 1:length(nodeIdx)
        inSum=sum(dGraph(:, nodeIdx(i)));
        if inSum ==0
            break;
        end
    end
    % remove this node and its out degree
    idx(iPass)=nodeIdx(i);
    nodeFlag(nodeIdx(i))=false;
    dGraph(nodeIdx(i), :)=0;
end
skeleton = skeleton(idx, idx);
end