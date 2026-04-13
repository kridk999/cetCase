import numpy as np
import pandas as pd


SIGNAL_Q_LOW = 0.10 
SIGNAL_Q_HIGH = 0.85

# load og prep data, beregn ratio, apply market stage
def load_and_prep_data(filepath: str = "assets/data.csv") -> pd.DataFrame:
    """load og prep data, beregn ratio, apply market stage"""
    df = pd.read_csv(filepath)

    df['date_time'] = pd.to_datetime(df['date_time'],utc=True).dt.tz_convert('CET')
    df = df.set_index('date_time')

    df["Hour"] = df.index.tz_convert("Europe/Copenhagen").hour + 1

    df["ratio_wind_to_load"] = (
        df["wind_forecast_dah_mw"] / df["consumption_forecast_dah_mw"]
    )

    def assign_market_stage(row):
        if pd.notna(row["intraday3"]) and pd.notna(row["intraday2"]):
            return "INTRADAY2"   # kan trade intraday2 -> intraday3
        if pd.notna(row["intraday2"]) and pd.notna(row["intraday1"]):
            return "INTRADAY1"   # kan trade intraday1 -> intraday2
        if pd.notna(row["intraday1"]) and pd.notna(row["spot"]):
            return "SPOT"        # kan trade spot -> intraday1
        return "NO_TRADE"

    df["market_stage"] = df.apply(assign_market_stage, axis=1)

    return df


def generate_signals(df: pd.DataFrame,) -> pd.DataFrame:
    """Generer signal baseret på ratio, undgår look-ahead bias ved at shift signalet 24 timer frem."""
    out = df.copy()

    temp_signal = pd.Series(0, index=out.index)
    temp_signal.loc[out["ratio_wind_to_load"] > SIGNAL_Q_HIGH] = 1
    temp_signal.loc[out["ratio_wind_to_load"] < SIGNAL_Q_LOW] = -1

    out["raw_signal"] = temp_signal.shift(24).fillna(0)

    return out


def execute_multi_leg_strategy(df):
    """Simulerer kun at kunne trade et bestemt leg afhængigt af time of day og market stage"""
    
    trade_records = []
    
    for i, row in df.iterrows():
        signal = row.get("raw_signal", 0)
        if signal == 0:
            continue

        hour = row.get("Hour")
        
        potential_legs = []
        

        if 1 <= hour <= 12:
            potential_legs.append(("spot", "intraday1", "SPOT_TO_ID1"))
            
        elif 13 <= hour <= 18:
            potential_legs.append(("intraday1", "intraday2", "ID1_TO_ID2"))
            
        elif 19 <= hour <= 24:
            potential_legs.append(("intraday2", "intraday3", "ID2_TO_ID3"))
        
        else:
            continue

        for entry_col, exit_col, leg_name in potential_legs:
            entry_px = row.get(entry_col, np.nan)
            exit_px = row.get(exit_col, np.nan)
            
            if pd.isna(entry_px) or pd.isna(exit_px):
                continue
                
            spread = exit_px - entry_px
            pnl_gross = spread * signal
            

            trade_records.append({
                "date_time": i,  
                "Hour": hour,
                "raw_signal": signal,
                "executed_leg": leg_name,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "execution_spread": spread,
                "realized_pnl_gross": pnl_gross,
                "realized_pnl_net": pnl_gross 
            })

            break # Kun én MwH per time 


    trades_df = pd.DataFrame(trade_records)
    

    trades_df["cumulative_pnl"] = trades_df["realized_pnl_net"].cumsum()
    
    return df, trades_df


def _build_metrics(trades: pd.DataFrame) -> dict:
    """Beregner metrics baseret på executed trades DataFrame"""
    if trades is None or trades.empty:
        return {
            "trade_count": 0,
            "win_rate": np.nan,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": np.nan,
        }

    pnl = trades["realized_pnl_net"].astype(float)
    total_pnl = float(pnl.sum())
    win_rate = float((pnl > 0).mean())

    pnl_std = float(pnl.std(ddof=0))
    sharpe_ratio = float(pnl.mean() / pnl_std) if pnl_std > 0 else np.nan

    running_max = trades["cumulative_pnl"].cummax()
    drawdown = running_max - trades["cumulative_pnl"]
    max_drawdown = float(drawdown.max())

    return {
        "trade_count": int(len(trades)),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }


def backtest(filepath: str = "assets/data.csv"):
    """Kører pipeline"""
    df = load_and_prep_data(filepath)
    df = generate_signals(df)
    
    df, trades_df = execute_multi_leg_strategy(df)

    metrics = _build_metrics(trades_df)

    return df, trades_df, metrics




if __name__ == "__main__":
    result_df, trades_df, metrics = backtest("assets/data.csv")
    print(metrics)
    trades_df.to_csv("assets/executed_trades.csv", index=True)

