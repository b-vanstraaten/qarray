import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, num2date
from matplotlib.finance import quotes_historical_yahoo, candlestick

# (Year, month, day) tuples suffice as aregs for quotes_historical_yahoo
date1 = (2004, 2, 1)
date2 = (2004, 4, 12)

mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
alldays = DayLocator()  # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # Eg, Jan 12
dayFormatter = DateFormatter('%d')  # Eg, 12

quotes = quotes_historical_yahoo('INTC', date1, date2)
if len(quotes) == 0:
    raise SystemExit

fig = plt.figure()
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(111)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)

candlestick(ax, quotes, width=0.6)

ax.xaxis_date()
ax.autoscale_view()
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


def on_move(event):
    ax = event.inaxes
    if ax is not None:
        # convert x y device coordinates to axes data coordinates
        date_ordinal, y = ax.transData.inverted().transform([event.x, event.y])

        # convert the numeric date into a datetime
        date = num2date(date_ordinal)

        # sort the quotes by their distance (in time) from the mouse position
        def sorter(quote):
            return abs(quote[0] - date_ordinal)

        quotes.sort(key=sorter)

        print
        'on date %s the nearest 3 openings were %s at %s respectively' % \
        (date,
         ', '.join([str(quote[1]) for quote in quotes[:3]]),
         ', '.join([str(num2date(quote[0])) for quote in quotes[:3]]))


on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()
