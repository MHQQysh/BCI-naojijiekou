function data = getData(subject_index_input, session_type_input)
% This function processes EEG data for a given subject and session type.
% It loads data, extracts trials, applies a band-pass filter, and saves
% the processed data and labels.
%
% Inputs:
%   subject_index_input: Integer, the subject index (e.g., 1-9)
%   session_type_input: Character, 'T' for training or 'E' for evaluation

    % Set your base directory for data and labels
    base_data_dir = 'C:\Users\13613\Desktop\Naoji\EEG-Conformer-main\preprocessing\BCICIV_2a_gdf\';
    base_label_dir = 'C:\Users\13613\Desktop\Naoji\EEG-Conformer-main\preprocessing\input\Labels\true_labels_2a\';
    base_output_dir = 'C:\Users\13613\Desktop\Naoji\EEG-Conformer-main\preprocessing\output\';

    % Construct file paths
    data_file = [base_data_dir, 'A0', num2str(subject_index_input), session_type_input, '.gdf'];
    label_file = [base_label_dir, 'A0', num2str(subject_index_input), session_type_input, '.mat'];

    % Load GDF data
    try
        [s, HDR] = sload(data_file);
    catch
        warning(['Could not load GDF file: ', data_file]);
        return; % Exit if file cannot be loaded
    end

    % Load Labels
    try
        load(label_file); % This loads 'classlabel' into workspace
        label_1 = classlabel;
    catch
        warning(['Could not load label file: ', label_file]);
        return; % Exit if file cannot be loaded
    end

    % Construct sample - data Section 1000*22*288
    Pos = HDR.EVENT.POS; % use POS to get trials
    Typ = HDR.EVENT.TYP;

    k = 0;
    data_1 = zeros(1000, 22, 288); % Pre-allocate for efficiency
    for j = 1:length(Typ)
        if Typ(j) == 768
            k = k + 1;
            % Ensure the indices do not go out of bounds
            start_idx = Pos(j) + 500;
            end_idx = Pos(j) + 1499;
            if end_idx <= size(s, 1)
                data_1(:,:,k) = s(start_idx:end_idx, 1:22);
            else
                warning(['Skipping trial ', num2str(j), ' due to out-of-bounds access.']);
                k = k - 1; % Decrement k if trial is skipped
            end
        end
    end
    
    % Resize data_1 if k is less than 288 (i.e., fewer trials found)
    if k < 288
        data_1 = data_1(:,:,1:k);
        % Also adjust label_1 if it corresponds to the number of trials
        if length(label_1) > k
             label_1 = label_1(1:k);
        end
    end


    % Wipe off NaN
    data_1(isnan(data_1)) = 0;

    %% Preprocessing - Band-pass filter
    fc = 250; % sampling rate
    Wl = 4; Wh = 40; % pass band
    Wn = [Wl*2 Wh*2]/fc;
    [b,a]=cheby2(6,60,Wn);
    
    % Apply filter to each trial
    for j = 1:size(data_1, 3)
        data_1(:,:,j) = filtfilt(b,a,data_1(:,:,j));
    end

    % Assign output variables
    data = data_1;
    label = label_1;
    
    % Save processed data
    saveDir = [base_output_dir, 'A0', num2str(subject_index_input), session_type_input, '.mat'];
    try
        save(saveDir,'data','label');
        disp(['Successfully processed and saved data for A0', num2str(subject_index_input), session_type_input]);
    catch
        warning(['Could not save data to: ', saveDir]);
    end

end