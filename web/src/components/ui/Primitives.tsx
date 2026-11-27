/** Reusable shadcn-style UI primitives aligned with Jace Studio Warm Archival Paper tokens. */

import type { ReactNode } from "react";
import { twMerge } from "tailwind-merge";

type ButtonVariant = "default" | "primary" | "fork" | "ghost";
type BadgeTone = "neutral" | "win" | "loss" | "warn" | "accent" | "fork";

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  default:
    "bg-[var(--surface)] text-[var(--text)] border-[var(--border)] hover:border-[var(--border-strong)]",
  primary: "bg-[var(--accent)] text-white border-transparent hover:opacity-90",
  fork: "bg-[var(--fork-fill)] text-white border-transparent hover:opacity-90",
  ghost:
    "bg-transparent text-[var(--text-2)] border-transparent hover:bg-[var(--surface-2)] hover:text-[var(--text)]",
};

const BADGE_STYLES: Record<BadgeTone, string> = {
  neutral: "bg-[var(--surface-2)] text-[var(--text-2)] border-[var(--border)]",
  win: "bg-[var(--win-soft)] text-[var(--win)] border-transparent",
  loss: "bg-[var(--loss-soft)] text-[var(--loss)] border-transparent",
  warn: "bg-[var(--warn-soft)] text-[var(--warn)] border-transparent",
  accent: "bg-[var(--accent-soft)] text-[var(--accent)] border-transparent",
  fork: "bg-[var(--fork-soft)] text-[var(--fork)] border-transparent",
};

interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: ButtonVariant;
  size?: "sm" | "md";
  disabled?: boolean;
  className?: string;
  title?: string;
  type?: "button" | "submit";
}

export function Button(props: ButtonProps): ReactNode {
  const variantStyle = BUTTON_STYLES[props.variant ?? "default"];
  const sizeStyle = props.size === "md" ? "px-3 py-1.5 text-xs" : "px-2.5 py-1 text-[11.5px]";
  return (
    <button
      type={props.type ?? "button"}
      onClick={props.onClick}
      disabled={props.disabled}
      title={props.title}
      className={twMerge(
        "inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed whitespace-nowrap",
        variantStyle,
        sizeStyle,
        props.className,
      )}
    >
      {props.children}
    </button>
  );
}

interface BadgeProps {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
}

export function Badge(props: BadgeProps): ReactNode {
  return (
    <span
      className={twMerge(
        "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] font-semibold border whitespace-nowrap",
        BADGE_STYLES[props.tone ?? "neutral"],
        props.className,
      )}
    >
      {props.children}
    </span>
  );
}

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  accent?: "default" | "fork";
}

interface SegmentedProps<T extends string> {
  options: ReadonlyArray<SegmentedOption<T>>;
  value: T;
  onChange: (value: T) => void;
  className?: string;
}

export function SegmentedControl<T extends string>(props: SegmentedProps<T>): ReactNode {
  return (
    <div
      className={twMerge(
        "inline-flex items-center rounded-md bg-[var(--surface-2)] p-0.5 border border-[var(--border)]",
        props.className,
      )}
    >
      {props.options.map((opt) => {
        const active = opt.value === props.value;
        const activeClass =
          opt.accent === "fork"
            ? "bg-[var(--fork-fill)] text-white font-semibold"
            : "bg-[var(--surface)] text-[var(--text)] font-semibold shadow-[var(--shadow-sm)]";
        return (
          <button
            key={opt.value}
            type="button"
            onClick={(): void => props.onChange(opt.value)}
            className={twMerge(
              "px-2.5 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer whitespace-nowrap",
              active ? activeClass : "text-[var(--text-2)] hover:text-[var(--text)]",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
