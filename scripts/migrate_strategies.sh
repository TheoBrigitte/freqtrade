#!/bin/bash
replace populate_buy_trend populate_entry_trend user_data/strategies/
replace populate_sell_trend populate_exit_trend user_data/strategies/
replace custom_sell custom_exit user_data/strategies/

replace buy_tag enter_tag user_data/strategies
replace sell_reason exit_reason user_data/strategies/

replace sell_profit_offset exit_profit_offset user_data/strategies/
replace ignore_roi_if_buy_signal ignore_roi_if_entry_signal user_data/strategies/
replace use_sell_signal use_exit_signal user_data/strategies/

replace "'buy':" "'entry':" user_data/strategies/
replace "'sell':" "'exit':" user_data/strategies/
replace "'emergencysell':" "'emergency_exit':" user_data/strategies/
replace "'forcebuy':" "'force_entry':" user_data/strategies/
replace "'forcesell':" "'force_exit':" user_data/strategies/
