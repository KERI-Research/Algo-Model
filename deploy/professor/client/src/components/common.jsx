/** Shared presentation primitives: brand mark, states, badges, tables. */

/**
 * MetaboGuard mark: a metabolic trace that steps out of range and returns.
 * Geometric, monochrome, legible at 20px and 200px, inherits currentColor.
 */
export const Logo = ({ size = 34, title = "MetaboGuard" }) => (
	<svg
		width={size}
		height={size}
		viewBox="0 0 40 40"
		role="img"
		aria-label={title}
		fill="none"
	>
		<rect
			x="1.25"
			y="1.25"
			width="37.5"
			height="37.5"
			rx="9"
			stroke="currentColor"
			strokeWidth="2"
		/>
		<path
			d="M8 26.5h4.4L16 13.5l3.6 9 2.6-4.4 3 6.4H32"
			stroke="currentColor"
			strokeWidth="2.2"
			strokeLinecap="round"
			strokeLinejoin="round"
		/>
		<circle cx="16" cy="13.5" r="2.6" fill="currentColor" />
	</svg>
);

export const Brand = ({ subtitle = "Research review dashboard" }) => (
	<div className="brand">
		<span style={{ color: "var(--accent)", display: "flex" }}>
			<Logo />
		</span>
		<span className="brand-text">
			<span className="brand-name">MetaboGuard</span>
			<span className="brand-sub">{subtitle}</span>
		</span>
	</div>
);

export const Notice = ({ kind = "info", title, children }) => (
	<div className="notice" data-kind={kind} role={kind === "blocked" ? "alert" : undefined}>
		{title ? <strong>{title}</strong> : null}
		{title ? " " : null}
		{children}
	</div>
);

export const Loading = ({ label = "Loading", rows = 4 }) => (
	<div className="card" aria-busy="true" aria-live="polite" data-testid="state-loading">
		<h4>{label}</h4>
		{Array.from({ length: rows }).map((_, index) => (
			<div className="skeleton" key={index} />
		))}
		<span className="visually-hidden">{label}...</span>
	</div>
);

export const ErrorState = ({ title = "Could not load this view", message, onRetry }) => (
	<div className="card" data-testid="state-error">
		<Notice kind="blocked" title={title}>
			{message}
		</Notice>
		{onRetry ? (
			<button type="button" className="btn btn-secondary" onClick={onRetry}>
				Try again
			</button>
		) : null}
	</div>
);

export const Empty = ({ children, testId = "state-empty" }) => (
	<p className="empty" data-testid={testId}>
		{children}
	</p>
);

export const TierBadge = ({ tier }) => (
	<span className="badge" data-tier={tier}>
		{String(tier || "unknown").replace(/_/g, " ")}
	</span>
);

export const Stat = ({ label, value, detail, state }) => (
	<div className="stat" data-state={state} data-testid={`stat-${label}`}>
		<div className="stat-label">{label}</div>
		<div className="stat-value">{value}</div>
		{detail ? <div className="stat-detail">{detail}</div> : null}
	</div>
);

export const DefinitionList = ({ items }) => (
	<dl className="dl">
		{items
			.filter(([, value]) => value !== undefined && value !== null && value !== "")
			.map(([term, value]) => (
				<div key={term} style={{ display: "contents" }}>
					<dt>{term}</dt>
					<dd>{value}</dd>
				</div>
			))}
	</dl>
);

export const formatNumber = (value, digits = 2) => {
	if (value === null || value === undefined || Number.isNaN(Number(value))) {
		return "n/a";
	}
	return Number(value).toLocaleString("en-GB", {
		minimumFractionDigits: digits,
		maximumFractionDigits: digits,
	});
};

export const formatPercent = (fraction, digits = 1) =>
	fraction === null || fraction === undefined
		? "n/a"
		: `${(Number(fraction) * 100).toFixed(digits)}%`;
