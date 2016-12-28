function [ FList ] = ReadFileNames(DataFolder)

if nargin < 1
DataFolder = uigetdir;
end

DirContents=dir(DataFolder);
FList=[];

if ~isunix
NameSeperator='\';
else isunix
NameSeperator='/';
end

extList={'jpg','peg','bmp','tif','iff','png'};

for i=1:numel(DirContents)
    if(~(strcmpi(DirContents(i).name,'.') || strcmpi(DirContents(i).name,'..')))
        if(~DirContents(i).isdir)
            extension=DirContents(i).name(end-2:end);
            if(numel(find(strcmpi(extension,extList)))~=0)
                FList=cat(1, FList,{[DataFolder,NameSeperator,DirContents(i).name]});
            end
        else
        getlist=ReadFileNames([DataFolder,NameSeperator,DirContents(i).name]);
        FList=cat(1, FList,getlist);
        end
    end
end

end
