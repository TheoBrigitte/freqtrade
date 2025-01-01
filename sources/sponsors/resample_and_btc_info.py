    def resampled_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe        

    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: "btc_" + s  if (not s in ignore_columns) else s, inplace=True)
        
        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: "btc_" + s if (not s in ignore_columns) else s, inplace=True)

        return dataframe   

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ''' 
        --> BTC informative (5m/1h)
        ___________________________________________________________________________________________
        '''
        btc_informative = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        btc_informative = self.base_tf_btc_indicators(btc_informative, metadata)
        dataframe = merge_informative_pair(dataframe, btc_informative, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [(s + "_" + self.timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_informative = self.dp.get_pair_dataframe("BTC/USDT", self.info_timeframe)
        btc_informative = self.info_tf_btc_indicators(btc_informative, metadata)
        dataframe = merge_informative_pair(dataframe, btc_informative, self.timeframe, self.info_timeframe, ffill=True)
        drop_columns = [(s + "_" + self.info_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        ''' 
        --> Informative timeframe
        ___________________________________________________________________________________________
        '''
        # populate informative indicators
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        # Merge informative into dataframe
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.info_timeframe, ffill=True)
        drop_columns = [(s + "_" + self.info_timeframe) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        ''' 
        --> Resampled to another timeframe
        ___________________________________________________________________________________________
        '''
        resampled = resample_to_interval(dataframe, timeframe_to_minutes(self.res_timeframe))
        resampled = self.resampled_tf_indicators(resampled, metadata)
        # Merge resampled info dataframe
        dataframe = resampled_merge(dataframe, resampled, fill_na=True)
        dataframe.rename(columns=lambda s: s+"_{}".format(self.res_timeframe) if "resample_" in s else s, inplace=True)
        dataframe.rename(columns=lambda s: s.replace("resample_{}_".format(self.res_timeframe.replace("m","")), ""), inplace=True)
        drop_columns = [(s + "_" + self.res_timeframe) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        ''' 
        --> The indicators for the normal (5m) timeframe
        ___________________________________________________________________________________________
        '''
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe