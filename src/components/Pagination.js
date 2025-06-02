import React from 'react';
import { useTranslation } from 'react-i18next';

const Pagination = ({ currentPage, totalPages, onPageChange, variant = 'default' }) => {
    const { t, i18n } = useTranslation();

    const themeClasses = {
        'etoile-du-sahel': {
            active: 'bg-red-600 text-white',
            hover: 'hover:bg-red-100 dark:hover:bg-red-900/30',
            text: 'text-red-600 dark:text-red-400'
        },
        'the-beautiful-game': {
            active: 'bg-green-600 text-white',
            hover: 'hover:bg-green-100 dark:hover:bg-green-900/30',
            text: 'text-green-600 dark:text-green-400'
        },
        'all-sports-hub': {
            active: 'bg-purple-600 text-white',
            hover: 'hover:bg-purple-100 dark:hover:bg-purple-900/30',
            text: 'text-purple-600 dark:text-purple-400'
        },
        archive: {
            active: 'bg-yellow-600 text-white',
            hover: 'hover:bg-yellow-100 dark:hover:bg-yellow-900/30',
            text: 'text-yellow-600 dark:text-yellow-400'
        },
        default: {
            active: 'bg-slate-600 text-white',
            hover: 'hover:bg-slate-100 dark:hover:bg-slate-900/30',
            text: 'text-slate-600 dark:text-slate-400'
        }
    };

    const theme = themeClasses[variant] || themeClasses.default;

    const renderPageNumbers = () => {
        const pages = [];
        const maxVisiblePages = 5;
        let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
        let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

        if (endPage - startPage + 1 < maxVisiblePages) {
            startPage = Math.max(1, endPage - maxVisiblePages + 1);
        }

        // First page
        if (startPage > 1) {
            pages.push(
                <button
                    key={1}
                    onClick={() => onPageChange(1)}
                    className={`px-3 py-1 rounded-md ${theme.hover} ${theme.text}`}
                >
                    1
                </button>
            );
            if (startPage > 2) {
                pages.push(
                    <span key="ellipsis1" className="px-2">
                        ...
                    </span>
                );
            }
        }

        // Page numbers
        for (let i = startPage; i <= endPage; i++) {
            pages.push(
                <button
                    key={i}
                    onClick={() => onPageChange(i)}
                    className={`px-3 py-1 rounded-md ${
                        i === currentPage
                            ? theme.active
                            : `${theme.hover} ${theme.text}`
                    }`}
                >
                    {i}
                </button>
            );
        }

        // Last page
        if (endPage < totalPages) {
            if (endPage < totalPages - 1) {
                pages.push(
                    <span key="ellipsis2" className="px-2">
                        ...
                    </span>
                );
            }
            pages.push(
                <button
                    key={totalPages}
                    onClick={() => onPageChange(totalPages)}
                    className={`px-3 py-1 rounded-md ${theme.hover} ${theme.text}`}
                >
                    {totalPages}
                </button>
            );
        }

        return pages;
    };

    if (totalPages <= 1) return null;

    return (
        <div className="flex items-center justify-center my-8 gap-x-3" dir="ltr">
            <button
                onClick={() => onPageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className={`p-2 rounded-md ${theme.hover} ${theme.text} disabled:opacity-50 disabled:cursor-not-allowed`}
                title={t('Previous page')}
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
            </button>

            <div className="flex items-center gap-x-2">
                {renderPageNumbers()}
            </div>

            <button
                onClick={() => onPageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className={`p-2 rounded-md ${theme.hover} ${theme.text} disabled:opacity-50 disabled:cursor-not-allowed`}
                title={t('Next page')}
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
            </button>
        </div>
    );
};

export default Pagination; 