% 示例：如何使用 sload 函数加载 GDF 格式的信号数据

% --- 1. 定义文件路径 ---

file_path = 'C:\Users\13613\Desktop\Naoji\EEG-Conformer-main\preprocessing\BCICIV_2a_gdf\A01E.gdf'; 
disp(['成功加载文件: ', file_path]);
% 示例路径，请替换为你的实际路径




% --- 2. 使用 sload 加载数据 ---
% [signal, H] = sload(FILENAME)
% signal: 包含加载的信号数据，通常是 [采样点数 x 通道数] 的矩阵


% H: 包含信号的头信息（header），是一个结构体
try
    [signal_data, header_info] = sload(file_path);



    % --- 3. 显示加载结果的基本信息 ---
    disp(['成功加载文件: ', file_path]);
    disp(' '); % 打印空行

    disp('--- 信号数据信息 ---');
    disp(['信号数据维度 (采样点数 x 通道数): ', mat2str(size(signal_data))]);
    disp(['信号数据类型: ', class(signal_data)]);


    disp(' '); % 打印空行
    disp('--- 头信息 (Header) 概览 ---');
    % 尝试显示一些常见的头信息字段
    if isfield(header_info, 'SampleRate')
        disp(['采样率 (SampleRate): ', num2str(header_info.SampleRate), ' Hz']);
    end
    if isfield(header_info, 'NChannels')
        disp(['通道数 (NChannels): ', num2str(header_info.NChannels)]);
    end
    if isfield(header_info, 'EVENT') && isfield(header_info.EVENT, 'TYP')
        disp(['事件类型数量 (Number of Event Types): ', num2str(length(unique(header_info.EVENT.TYP)))]);
        disp(['部分事件类型 (First 5 Event Types): ', mat2str(header_info.EVENT.TYP(1:min(5, end)))]);
    end
    if isfield(header_info, 'Label')
        disp('通道标签 (Channel Labels):');
        disp(header_info.Label(1:min(5, end), :)); % 显示前5个通道标签
    end



    % --- 4. 进一步示例：加载特定通道 ---
    % 假设我们只想加载通道1和通道3
    selected_channels = [1, 3];
    disp(' ');
    disp(['--- 再次加载：仅读取通道 ', mat2str(selected_channels), ' ---']);
    [signal_subset, header_subset] = sload(file_path, selected_channels);
    disp(['子集信号数据维度: ', mat2str(size(signal_subset))]);
    disp(['子集信号数据中的通道数: ', num2str(size(signal_subset, 2))]);






    % --- 5. 进一步示例：使用可选属性 'UCAL' ---
    % 'UCAL' 表示加载未校准（未缩放）的原始数据
    disp(' ');
    disp('--- 再次加载：使用 ''UCAL'' 属性 (加载未校准数据) ---');
    [signal_uncal, header_uncal] = sload(file_path, 'UCAL', 'On');
    disp(['未校准信号数据维度: ', mat2str(size(signal_uncal))]);
    disp(['未校准信号数据类型: ', class(signal_uncal)]);
    % 注意：未校准数据通常是整数类型，需要手动缩放

catch ME
    % 错误处理
    warning(['加载文件时发生错误: ', ME.message]);
    disp('请检查：');
    disp('1. Biosig 工具箱是否已正确安装并添加到 MATLAB 路径。');
    disp('2. 文件路径是否正确，且文件是否存在。');
    disp('3. 文件是否为 sload 支持的格式 (例如 .gdf)。');
end
