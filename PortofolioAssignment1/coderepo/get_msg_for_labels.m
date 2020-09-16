function [ msg ] = get_msg_for_labels(l)
%     Convert a one-dimensional label array to a message-string.
% 
%     Args:
%         l (1D array): Predicted (binary) labels which should be converted to a
%                       message. The first eight elements are assumed to be the
%                       8-bit ASCII representation of the first letter in the
%                       message, the next eight elements are assumed to represent
%                       the next letter, and so on. length(l) must be a multiple of 8
%     Returns:
%         msg (string): Decoded message.

% Divide into vectors with 8 "bits"
n_bits = 8;
l = reshape(l,n_bits,[]);
l = l.'; % transpose

% Build a vector with binary positional values (...,64,32,16,8,4,2,1)
bin_pos_vec = 2*ones(1,n_bits);
bin_pos_vec = bin_pos_vec.^((n_bits-1):-1:0); % MSB
bin_pos_vec = bin_pos_vec.';

% Get decimal values of message
msg_dec = l * bin_pos_vec;
msg_dec = msg_dec.';

% Convert to ASCII
msg = char(msg_dec);

